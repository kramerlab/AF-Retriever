"""Lightweight parsing of Cypher query fragments into ``Triplet`` objects.

This is *not* a full Cypher parser. It recognises a few constructs with regexes
and ignores everything else (``RETURN``, ``ORDER BY``, functions, ...).

Usage::

    triplets, symbols, warnings = parse_cypher_to_triplets(
        query, rel_dict, properties_dict, node_type_list,
        skip_triplets_w_invalid_rel_type=True, skip_symbs_w_invalid_type=True)
    parse_conditions_from_cypher(query, symbols, properties_dict)

Supported syntax
----------------
Relationships (``parse_cypher_to_triplets``)
    * ``(head)-[:REL]->(tail)`` and ``(tail)<-[:REL]-(head)``; the arrow always
      yields ``head -REL-> tail``.
    * Chains and mixed directions, each hop becoming its own triplet, e.g.
      ``(a)-[:R]->(b)<-[:R]-(c)`` gives ``a -> b`` and ``c -> b``.
      Results are ordered left-pointing arrows first, then right-pointing ones,
      not by position in the query.
    * Self-loops such as ``(a)-[:R]->(a)``; both ends share one ``TripletEnd``.
    * Relation names may be translated through ``rel_dict`` (query name ->
      database name); its values are the set of valid relations.

Nodes (inside the parentheses)
    * ``(name:Type)``, ``(name:Type {prop: 'value'})`` (a space before ``{`` is
      optional) and ``(:Type {prop: 'value'})`` for a node without a name, which
      gets a generated name ``untitled_const<N>``.
    * ``(name)``: a node without a type. Its type may be given by any other
      mention of the same variable, anywhere in the query (see
      ``skip_symbs_w_invalid_type`` below).
    * A node that has a ``name`` or ``title`` property is flagged ``is_constant``.
    * A node pattern with a property that appears on its own (not in a
      relationship) is also returned as a symbol.

Conditions (``parse_conditions_from_cypher``): ``var.prop <op> value``
    * ``=``, ``=~`` and ``CONTAINS`` with a quoted value or a bare word/number.
    * ``>``, ``>=``, ``<``, ``<=`` with integers, decimals (``3.5``) and
      negatives (``-5``). The operator is kept as a prefix of the stored value,
      e.g. ``">=5"``.
    * ``IN [...]`` with quoted items (``['a', 'b']``) or unquoted items
      (``[aspirin, $ids]``). Empty lists are ignored.
    * Several values/conditions on the same property are joined with ``"; "``.
    * The case-insensitive marker ``(?i)`` is stripped from values.

Not supported
    * ``<>`` (not equal) and other operators: such conditions are silently
      ignored.
    * Variable-length paths (``[:R*1..3]``), relationship variables or properties
      (``[x:R {w: 1}]``), multiple labels (``(a:A:B)``) and the bare ``--``
      connector: these are not understood. Depending on the ``skip_*`` flags the
      triplet is dropped, or kept with relation ``"*"`` / node type ``None``.

Undirected relationships
    ``(a)-[:R]-(b)`` (no arrowhead) is accepted but treated as directed
    ``a -R-> b``, i.e. as if it were written ``(a)-[:R]->(b)``.

Behaviour notes
---------------
``skip_symbs_w_invalid_type``
    A symbol is dropped, together with every triplet using it, only if it never
    receives a valid type. The check runs after the whole query has been parsed,
    so ``(a)-[:R]->(b:B)`` followed by ``(a:A)-[:S]->(c:C)`` keeps both triplets.
    A node with an explicit but unknown type (``(a:Foo)``) is rejected at once,
    and so is the triplet it is in. The triplet's other, valid end is not lost:
    it is still registered, so a type or property seen there is available to
    later mentions of the same variable. For example, in
    ``(a:Foo)-[:R]->(y:B)  (c:C)-[:S]->(y)`` the first triplet is dropped, but
    ``y`` keeps its type ``B`` and the second triplet is kept.

``skip_triplets_w_invalid_rel_type``
    If True, triplets with an unknown relation are dropped; otherwise the
    relation is replaced by ``"*"``.

Masking of quoted literals
    By default both parsing functions mask ``( ) [ ] { } :`` inside quoted
    values before matching (see :func:`mask_structural_chars_in_quotes`), so a
    value such as ``'2-(4-chlorophenyl)ethanol'`` is not mistaken for Cypher
    syntax, and restore them in the values they store. Callers do not need to
    mask or unmask anything. Pass ``mask_quoted_literals=False`` to turn this off.
"""

import ast
import regex as re

from triplet import TripletEnd, Triplet


# ---------------------------------------------------------------------------
# Masking of structural characters inside quoted literals
# ---------------------------------------------------------------------------

# Private-use Unicode code points won't collide with real input
_MASK = {'(': '\uE000', ')': '\uE001', '[': '\uE002', ']': '\uE003',
         '{': '\uE004', '}': '\uE005', ':': '\uE006'}
_UNMASK = {v: k for k, v in _MASK.items()}

_QUOTED_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')


def mask_structural_chars_in_quotes(cypher_str: str) -> str:
    """Replace ``( ) [ ] { } :`` found INSIDE quoted literals with placeholders.

    This stops the structural regexes below from misparsing literal values
    (e.g. chemical names such as ``'2-(4-chlorophenyl)ethanol'``) as Cypher
    syntax. The quote characters themselves are left in place. Use
    :func:`unmask_structural_chars` on extracted values to restore them.
    """
    def _replace(match: re.Match) -> str:
        literal = match.group(0)
        for char, placeholder in _MASK.items():
            literal = literal.replace(char, placeholder)
        return literal

    return _QUOTED_LITERAL.sub(_replace, cypher_str)


def unmask_structural_chars(value: str) -> str:
    """Inverse of :func:`mask_structural_chars_in_quotes`."""
    for placeholder, char in _UNMASK.items():
        value = value.replace(placeholder, char)
    return value


# ---------------------------------------------------------------------------
# WHERE-style conditions
# ---------------------------------------------------------------------------

# Groups: 1 variable, 2 property, 3 operator, 4 quoted value, 5 bare word/number
#         (integers, decimals and negatives included),
#         6 list literal (only for IN). Groups 3-5 and group 6 are mutually exclusive.
_CONDITION_PATTERN = re.compile(
    r'(\w+)\.(\w+)\s*'
    r'(?:(=(?:~?)|CONTAINS|<=|>=|<|>)\s*(?:[\'"]([^\'"]*)[\'"]|(-?\d+(?:\.\d+)?\b|\w+))'
    r'|IN\s*(\[[^\]]*\]))')

_COMPARISON_OPERATORS = ("<", ">", "<=", ">=")

# One item of a list literal: 'quoted', "quoted" or a bare token (aspirin, 42, $param)
_LIST_ITEM = re.compile(r"""'([^']*)'|"([^"]*)"|([^\s,\[\]]+)""")


def _parse_in_list(list_literal: str) -> list[str]:
    """Turn the ``[...]`` of an ``IN`` condition into a list of strings.

    Tries ``ast.literal_eval`` first (handles escapes in quoted items). If that
    fails, e.g. because the items are unquoted words like ``[aspirin, ibuprofen]``
    or parameters like ``[$ids]``, the items are split out with a tolerant
    tokenizer instead, so no value is lost.
    """
    try:
        return [str(v) for v in ast.literal_eval(list_literal)]
    except (ValueError, SyntaxError):
        return [quoted1 or quoted2 or bare
                for quoted1, quoted2, bare in _LIST_ITEM.findall(list_literal)]


def parse_conditions_from_cypher(cypher_query: str,
                                 triplet_ends: dict[str, TripletEnd],
                                 properties_dict: dict[str, str],
                                 mask_quoted_literals: bool = True) -> None:
    """Attach WHERE-style conditions to the matching ``TripletEnd`` objects (in place).

    For every ``var.prop <op> value`` or ``var.prop IN [...]`` found in the
    query whose ``var`` is a key of ``triplet_ends``:

    * the value(s) are stored in ``triplet_ends[var].properties[prop]`` as a
      ``"; "``-joined string (appended if the property already exists),
    * comparison operators ``< > <= >=`` are kept as a prefix of the value
      (e.g. ``">5"``),
    * a case-insensitive marker ``(?i)`` is stripped from values,
    * ``IN`` lists may contain quoted or unquoted items; empty lists are ignored,
    * the end is flagged ``is_constant`` once it has a ``name`` or ``title``.

    Args:
        cypher_query: Query text.
        triplet_ends: Variable name -> ``TripletEnd``. Unknown variables are ignored.
        properties_dict: Maps property names used in the query to the names
            used in the database.
        mask_quoted_literals: Mask ``( ) [ ] { } :`` inside quoted values before
            parsing (default True). Values are always unmasked again before
            being stored. Set to False only if the query is already masked or
            known to contain no such characters in literals.
    """
    if mask_quoted_literals:
        cypher_query = mask_structural_chars_in_quotes(cypher_query)

    def clean(value: str) -> str:
        # Unmask first: "(?i)" itself contains parentheses that were masked.
        return unmask_structural_chars(value).replace("(?i)", "")

    for (var_name, var_property, operator,
         quoted_value, bare_value, list_literal) in _CONDITION_PATTERN.findall(cypher_query):

        value = quoted_value + bare_value
        if operator in _COMPARISON_OPERATORS:
            value = operator + value

        if list_literal:
            values = [clean(v) for v in _parse_in_list(list_literal)]
        else:
            values = [clean(value)]

        end = triplet_ends.get(var_name)
        if end is None:
            continue

        if values:  # an empty list (IN []) contributes nothing
            prop = properties_dict.get(var_property, var_property)
            joined = "; ".join(values)
            if prop in end.properties:
                end.properties[prop] += "; " + joined
            else:
                end.properties[prop] = joined

        if "title" in end.properties or "name" in end.properties:
            end.is_constant = True


# ---------------------------------------------------------------------------
# Relationship / node patterns
# ---------------------------------------------------------------------------

# (A)<-[...REL]-(B)   i.e. B -REL-> A.   Groups: A, REL, B
_PATTERN_LEFT = re.compile(r'\(([^)]+)\)<-\[[^:\]]*\:?([^\]]+)\]-\(([^)]+)\)')
# (A)-[...REL]->(B)   i.e. A -REL-> B.   Groups: A, REL, B
_PATTERN_RIGHT = re.compile(r'\(([^)]+)\)-\[[^:\]]*\:?([^\]]+)\]->?\(([^)]+)\)')
# A node that carries a property, e.g. (x:Drug {name: 'aspirin'})
_PATTERN_SYMBOL_W_PROPERTY_ONLY = re.compile(
    r'\(\w+\:?[^\s\)]+\s*\{\w+\s*\:\s*[\'"][^\'"]+[\'"]\}\)')

# The content of one pair of parentheses, either
#   name:Type  or  name:Type{prop: 'value'}   (name may be empty)
#   name                                       (name only, no type)
# Groups: name, type, property, property value, name-only
_NODE_PATTERN = re.compile(
    r'(?:(\w*)\:([^\s\)\{]+)\s*(?:\{(\w+)\s*\:\s*[\'"]([^\'"]*)[\'"]\})?)|'
    r'(?:(\w+))')


def parse_cypher_to_triplets(
        cypher_query: str,
        rel_dict: dict[str, str],
        properties_dict: dict[str, str],
        node_type_list: list[str],
        skip_triplets_w_invalid_rel_type: bool,
        skip_symbs_w_invalid_type: bool,
        mask_quoted_literals: bool = True) -> tuple[list[Triplet], dict[str, TripletEnd], str]:
    """Extract triplets and the node symbols they reference from a Cypher query.

    Args:
        cypher_query: Query text.
        rel_dict: Maps relation names as written in the query to relation names
            in the database. Its *values* are the set of valid relations.
        properties_dict: Maps property names used in the query to database names.
        node_type_list: All valid node types.
        skip_triplets_w_invalid_rel_type: If True, triplets whose relation is
            not valid are dropped; otherwise the relation becomes ``"*"``.
        skip_symbs_w_invalid_type: If True, symbols without a valid node type
            are dropped together with every triplet that uses them; otherwise
            their ``node_type`` is left as ``None``.

            A node may legitimately be written without a type (``(a)``) and be
            typed elsewhere in the query (``(a:Drug)``). Because of that, this
            filter is applied only AFTER the whole query has been parsed, and
            a symbol is dropped only if no valid type was ever seen for it.
            A node with an explicit but unknown type (``(a:Foo)``) is rejected
            immediately.
        mask_quoted_literals: Mask ``( ) [ ] { } :`` inside quoted values before
            parsing (default True), so values such as ``'foo (bar)'`` are not
            mistaken for Cypher syntax. Property values are unmasked again
            before being stored. Set to False only if the query is already
            masked or known to contain no such characters in literals.

    Returns:
        ``(triplets, symbols, warnings)``. ``symbols`` maps variable names to
        ``TripletEnd`` objects; nodes without a name get a generated name
        ``untitled_const<N>``. ``warnings`` is a ``"; "``-joined string
        describing what was removed (empty if nothing was).
    """
    warnings: list[str] = []
    if mask_quoted_literals:
        cypher_query = mask_structural_chars_in_quotes(cypher_query)

    # Every valid node is registered here as soon as it is parsed, even if the
    # triplet it appears in is later rejected. That way information seen in a
    # rejected triplet (e.g. the type in "(y:gene/protein)") is still available
    # to other mentions of the same variable ("(y)").
    symbols: dict[str, TripletEnd] = {}
    # Keep head/tail next to each triplet so the final type filter does not
    # depend on the internals of Triplet.
    accepted: list[tuple[Triplet, TripletEnd, TripletEnd]] = []
    valid_relations = set(rel_dict.values())

    def add_symbol(node: str) -> TripletEnd | None:
        """Parse one node string, register it in ``symbols`` and return it.

        An already registered variable is reused and enriched with the new
        type/property. Returns None if the node cannot be parsed, or if it has
        an explicit invalid type and ``skip_symbs_w_invalid_type`` is set.
        """
        node_matches = _NODE_PATTERN.findall(node.strip())
        if len(node_matches) != 1:
            return None
        var_name, n_type, n_property, n_property_value, name_only = node_matches[0]
        n_property_value = unmask_structural_chars(n_property_value)

        if name_only:
            # "(a)": no type information here. Don't judge it now - the type
            # may be given by another mention of the same variable.
            var_name = name_only
            n_type = None
        elif n_type not in node_type_list:
            if skip_symbs_w_invalid_type:
                warnings.append("Removed invalid symbol from Cypher query")
                return None
            n_type = None

        if var_name == "":  # constant without a name
            idx = len(symbols)
            while f"untitled_const{idx}" in symbols:
                idx += 1
            var_name = f"untitled_const{idx}"

        if var_name not in symbols:
            symbols[var_name] = TripletEnd(var_name, n_type, is_constant=False)
        symbol = symbols[var_name]
        if symbol.node_type is None:
            symbol.node_type = n_type

        if n_property != "":
            n_property = properties_dict.get(n_property, n_property)
            if n_property in symbol.properties:
                if n_property_value not in symbol.properties[n_property]:
                    symbol.properties[n_property] += "; " + n_property_value
            else:
                symbol.properties[n_property] = n_property_value
            if n_property in ("name", "title"):
                symbol.is_constant = True
        return symbol

    def add_triplet(head_str: str, relation: str, tail_str: str) -> None:
        """Build ``head -relation-> tail`` and record it if it is acceptable."""
        # Parse both ends first: a valid end is registered even if the other
        # end (or the relation) makes this triplet unacceptable.
        head = add_symbol(head_str)
        tail = add_symbol(tail_str)
        if head is None or tail is None:
            return

        relation = relation.split(":")[-1]
        relation = rel_dict.get(relation, relation)  # translate to database name
        if relation not in valid_relations:
            if skip_triplets_w_invalid_rel_type:
                warnings.append("Removed invalid triplet from Cypher query")
                return
            relation = "*"

        accepted.append((Triplet(head, relation, tail), head, tail))

    # (tail)<-[REL]-(head)
    for tail_str, relation, head_str in _PATTERN_LEFT.findall(cypher_query, overlapped=True):
        add_triplet(head_str.strip(), relation.strip(), tail_str.strip())
    # (head)-[REL]->(tail)
    for head_str, relation, tail_str in _PATTERN_RIGHT.findall(cypher_query, overlapped=True):
        add_triplet(head_str.strip(), relation.strip(), tail_str.strip())
    # Nodes that appear on their own with a property
    standalone_names: set[str] = set()
    for node_str in _PATTERN_SYMBOL_W_PROPERTY_ONLY.findall(cypher_query):
        symbol = add_symbol(node_str)
        if symbol is not None:
            standalone_names.add(symbol.name)

    # Deferred type check: now that every mention has been merged, drop
    # triplets that still have an end without a valid type.
    num_accepted = len(accepted)
    referenced = {end.name for _, head, tail in accepted for end in (head, tail)}
    referenced |= standalone_names

    if skip_symbs_w_invalid_type:
        accepted = [(t, head, tail) for t, head, tail in accepted
                    if head.node_type is not None and tail.node_type is not None]

    if len(accepted) != num_accepted:
        warnings.append("Removed invalid triplets from Cypher query")

    # Keep only symbols that are still used by a triplet or stand on their own,
    # so symbols of dropped triplets don't leak into the result.
    used_names = {end.name for _, head, tail in accepted for end in (head, tail)}
    used_names |= standalone_names

    if skip_symbs_w_invalid_type and any(symbols[n].node_type is None for n in referenced):
        warnings.append("Removed invalid symbols from Cypher query")

    symbols = {name: s for name, s in symbols.items()
               if name in used_names
               and (s.node_type is not None or not skip_symbs_w_invalid_type)}
    triplets = [t for t, _, _ in accepted]

    return triplets, symbols, "; ".join(dict.fromkeys(warnings))
