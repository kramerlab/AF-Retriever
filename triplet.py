class TripletEnd:
    def __init__(self, name: str, node_type: str, is_constant: bool, candidates: set = None):
        self.name = name
        self.node_type = node_type
        self.is_constant = is_constant
        self.candidates = candidates
        self.properties = {}

    def __iter__(self):
        return self.candidates.__iter__()

    def __repr__(self):
        num_candidates = len(self.candidates) if self.candidates is not None else "na"
        return f"{self.get_uid()}, {self.is_constant=}, {self.properties=}, {num_candidates=}"

    def __str__(self):
        return self.__repr__()


    def get_uid(self):
        return self.name + "::" + str(self.node_type)

    def intersection_update(self, node_ids):
        self.candidates = set(node_ids).intersection(self.candidates)


class Triplet:
    def __init__(self, head: TripletEnd, edge: str, tail: TripletEnd):
        self.h = head
        self.e = edge
        self.t = tail

    def __repr__(self):
        return f"{self.h.get_uid()} -> {self.e} -> {self.t.get_uid()}"

    def str_extended(self):
        return f"{self.h.properties} -> {self.e} -> {self.t.properties}"

    def __str__(self):
        return self.__repr__()