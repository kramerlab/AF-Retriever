import ast
import regex as re

from triplet import TripletEnd, Triplet


def parse_conditions_from_cypher(cypher_query: str, triplet_ends: dict[str, TripletEnd],
                                 properties_dict: dict[str,str]):
    equals_pattern = re.compile(r'(\w+)\.(\w+)\s*(?:(=(?:~?)|CONTAINS|<|>|<=|>=)\s*(?:[\'"]([^\'"]*)[\'"]|(\w+))|IN\s*(\[[^\]]*\]))')

    atoms = re.findall(equals_pattern, cypher_query)
    for atom in atoms:
        var_name, var_property, var_equality, var_value, var_value_word, var_values = atom
        var_value += var_value_word
        if var_equality == "<" or var_equality == ">" or var_equality == "<=" or var_equality == ">=":
            var_value = var_equality + var_value
        try:
            var_values = ast.literal_eval(var_values)
            if var_value != "":
                var_values.append(var_value.replace("(?i)", ""))
        except (ValueError, SyntaxError):
            var_values = [var_value.replace("(?i)", "")]

        if var_name in triplet_ends:
            if var_property != "":
                if var_property in properties_dict:
                    var_property = properties_dict[var_property]
                if var_property in triplet_ends[var_name].properties:
                    triplet_ends[var_name].properties[var_property] += "; " + "; ".join(var_values)
                else:
                    triplet_ends[var_name].properties[var_property] = "; ".join(var_values)

            if "title" in triplet_ends[var_name].properties or "name" in triplet_ends[var_name].properties:
                triplet_ends[var_name].is_constant = True


def parse_cypher_to_triplets(cypher_query: str, rel_dict: dict[str, str], properties_dict: dict[str,str],
                             node_type_list: list[str], skip_triplets_w_invalid_rel_type: bool,
                             skip_symbs_w_invalid_type: bool) -> [list[Triplet], dict[str, TripletEnd]]:
    # Regular expression to match patterns like (HEAD)-[RELATION]->(TAIL) or (HEAD)<-[RELATION]-(TAIL)
    symbols = {}

    pattern_left = r'\(([^)]+)\)<-\[[^:\]]*\:?([^\]]+)\]-\(([^)]+)\)'
    pattern_right = r'\(([^)]+)\)-\[[^:\]]*\:?([^\]]+)\]->?\(([^)]+)\)'
    pattern_symbol_w_property_only = r'\(\w+\:?[^\s\)]+\s*\{\w+\s*\:\s*[\'"][^\'"]+[\'"]\}\)'

    node_pattern = (r'(?:(\w*)\:([^\s\)]+)\s*(?:\{(\w+)\s*\:\s*[\'"]([^\'"]*)[\'"]\})?)|'
                    r'(?:(\w+))') # [name]:[type] or [name]:[type]{n_property:n_value}

    # Convert matches to list of triplets
    triplets = []

    def add_symbol(node: str, idx: int):
        node_matches = re.findall(node_pattern, node.strip())
        if len(node_matches) != 1:
            return None
        var_name, n_type, n_property, n_property_value, var_name_only = node_matches[0]
        if n_type not in node_type_list:
            if skip_symbs_w_invalid_type:
                return None
            else:
                n_type = None
        if var_name_only != "":  # no node_properties
            var_name = var_name_only
        if var_name == "":  # constant without name
            var_name = "untitled_const" + str(idx)
        if var_name in symbols:  # loading already existing variable
            symbol = symbols[var_name]
        else:
            symbol = TripletEnd(var_name, n_type, is_constant=False)
        if symbol.node_type is None:
            symbol.node_type = n_type

        if n_property != "":
            if n_property in properties_dict:
                n_property = properties_dict[n_property]
            if n_property in symbol.properties:
                if n_property_value not in symbol.properties[n_property]:
                    symbol.properties[n_property] += "; " + n_property_value
            else:
                symbol.properties[n_property] = n_property_value
            if n_property in ["name", "title"]:
                symbol.is_constant = True
        return symbol

    def add_triplet(tail, relation, head):
        head = add_symbol(head, len(symbols))
        tail = add_symbol(tail, len(symbols) + 1)
        if head is None or tail is None:
            return None
        relation = relation.split(":")[-1]
        if relation in rel_dict:
            relation = rel_dict[relation]  # exchange by relation name in database
        if relation not in rel_dict.values():
            if skip_triplets_w_invalid_rel_type:
                return None
            else:
                relation = "*"
        symbols[head.name] = head
        symbols[tail.name] = tail
        return Triplet(head, relation, tail)

    # Find all matches in the Cypher query
    for match in re.findall(pattern_left, cypher_query, overlapped=True):
        tail, relation, head = match[2].strip(), match[1].strip(), match[0].strip()
        triplet = add_triplet(head, relation, tail)
        if triplet is not None:
            triplets.append(triplet)
    for match in re.findall(pattern_right, cypher_query, overlapped=True):
        tail, relation, head = match[0].strip(), match[1].strip(), match[2].strip()
        triplet = add_triplet(head, relation, tail)
        if triplet is not None:
            triplets.append(triplet)
    symbols_w_props = re.findall(pattern_symbol_w_property_only, cypher_query)
    for symbol in symbols_w_props:
        symbol = add_symbol(symbol, len(symbols))
        if symbol is not None:
            symbols[symbol.name] = symbol
    return triplets, symbols

