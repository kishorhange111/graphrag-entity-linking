"""A synthetic knowledge world: KB entities, documents with noisy mentions, multi-hop questions.

Entities
    people (first/last name, research field, birthplace, employer), organisations
    (two-word name, acronym, short alias, sector, HQ city, founder, optional acquirer),
    cities. Names are drawn from small pools, so **homonyms happen naturally**
    (about 1 in 7 people shares a full name), acronyms collide, and surname-only
    mentions are highly ambiguous.

Documents
    person bios, organisation profiles, acquisition notes and distractor news
    ("X gave a talk in Lyon") that mention entities without stating a relation.
    Later mentions use short forms ("Weber", "Dr. Weber", "NVL"). Every mention has
    a gold entity link, used only for grading.

Questions
    1-, 2- and 3-hop relation paths from a question entity, e.g.
    ``works_at → acquired_by → hq_in``, with the supporting fact for each hop.

The KB available to the entity linker holds names, aliases and *descriptions* (a
person's research field, an organisation's sector), like a Wikidata-style
catalogue. It does **not** hold the relations the questions ask about; those must
come from the documents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

FIRST = (
    "Anna Ben Carla David Elena Farid Grace Hiro Ines Jonas Kavya Liam Mona Noah Olga Pavel "
    "Quinn Rosa Sami Tara Umar Vera Wei Ximena Yusuf Zoe Arjun Bianca Chen Dmitri Esra Felix "
    "Greta Hassan Ivy Jamal Keiko Lars Maya Nikhil"
).split()
LAST = (
    "Weber Okafor Rossi Tanaka Novak Silva Haddad Kim Larsen Mehta Fischer Dubois Costa Ivanov "
    "Sato Nguyen Kowalski Moreau Petrov Schmidt Garcia Ali Brown Jensen Romano Yilmaz Bauer "
    "Kaur Lindqvist Herrera Osei Varga Popescu Nakamura Fernandes Hansen Kovacs Morales Singh "
    "Walsh Becker Lund Abbas Torres Volkov Ito Reyes Hoffmann Patel Quist"
).split()
ORG_A = (
    "Northvale Brightwater Cobalt Delphi Ember Fjord Granite Helix Ionic Juniper Kestrel Lumen "
    "Meridian Nimbus Onyx Pinnacle Quanta Redwood Sable Tidal Umbra Vertex Willow Xenon Yarrow "
    "Zephyr Aurora Basalt Cinder Drift Echo Falcon Glacier Harbor Indigo Jade Krypton Lattice "
    "Mosaic Nova Orbit Prism Quartz Ripple Summit Talon Unity Vortex Wren"
).split()
ORG_B = "Labs Systems Bio Robotics Analytics Dynamics Networks Health".split()
CITIES = (
    "Lyon Pune Austin Osaka Leeds Turin Porto Graz Haifa Aarhus Kyoto Bergen Tartu Cork Ghent "
    "Brno Malmo Basel Quito Perth Accra Lille Bilbao Krakow Seville Dresden Utrecht Tampere Split "
    "Bologna Nantes Leipzig Mysore Sendai Durban Hobart Tromso Galway Gdansk Nagoya"
).split()
FIELDS = (
    "robotics genomics cryptography photonics neuroscience climate materials linguistics "
    "optimization epidemiology astrophysics acoustics"
).split()
SECTORS = "biotech logistics fintech energy semiconductor aerospace retail insurance gaming".split()


@dataclass
class Entity:
    id: str
    kind: str  # person | org | city
    name: str
    aliases: list[str]
    description: str = ""
    attrs: dict = field(default_factory=dict)


@dataclass
class Mention:
    start: int
    end: int
    entity: str  # gold link


@dataclass
class Doc:
    id: int
    text: str
    mentions: list[Mention]
    facts: list[tuple[str, str, str]]  # gold (subj, rel, obj) the doc states
    topic: str | None = None


@dataclass
class Question:
    text: str
    hops: int
    path: list[str]  # relation path
    start: str  # gold question entity
    chain: list[str]  # gold entities after each hop
    support: list[tuple[str, str, str]]  # gold fact per hop

    @property
    def answer(self) -> str:
        return self.chain[-1]


@dataclass
class World:
    entities: dict[str, Entity]
    docs: list[Doc]
    questions: list[Question]
    triples: set[tuple[str, str, str]]

    def by_kind(self, kind: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.kind == kind]


def an(word: str) -> str:
    return f"an {word}" if word[0] in "aeiou" else f"a {word}"


class _DocBuilder:
    def __init__(self, did: int, topic: str | None = None):
        self.did, self.topic = did, topic
        self.parts: list[str] = []
        self.mentions: list[Mention] = []
        self.facts: list[tuple[str, str, str]] = []
        self.len = 0

    def text(self, s: str) -> _DocBuilder:
        self.parts.append(s)
        self.len += len(s)
        return self

    def ent(self, surface: str, eid: str) -> _DocBuilder:
        self.mentions.append(Mention(self.len, self.len + len(surface), eid))
        return self.text(surface)

    def build(self) -> Doc:
        return Doc(self.did, "".join(self.parts), self.mentions, self.facts, self.topic)


def make_world(n_people: int = 400, n_orgs: int = 100, n_distractors: int = 250, seed: int = 0):
    rng = random.Random(seed)
    E: dict[str, Entity] = {}
    cities = [Entity(f"C{i}", "city", c, [c]) for i, c in enumerate(CITIES)]
    for c in cities:
        E[c.id] = c

    org_names: set[str] = set()
    orgs: list[Entity] = []
    while len(orgs) < n_orgs:
        a, b = rng.choice(ORG_A), rng.choice(ORG_B)
        if f"{a} {b}" in org_names:
            continue
        org_names.add(f"{a} {b}")
        sector = rng.choice(SECTORS)
        e = Entity(
            f"O{len(orgs)}",
            "org",
            f"{a} {b}",
            [f"{a} {b}", a, a[0] + b[0]],
            f"{a} {b} ({a[0] + b[0]}), {an(sector)} company",
            {"sector": sector, "hq": rng.choice(cities).id},
        )
        orgs.append(e)
    people: list[Entity] = []
    for i in range(n_people):
        f, l_ = rng.choice(FIRST), rng.choice(LAST)
        fld = rng.choice(FIELDS)
        e = Entity(
            f"P{i}",
            "person",
            f"{f} {l_}",
            [f"{f} {l_}", l_, f"{f[0]}. {l_}", f"Dr. {l_}"],
            f"{f} {l_}, {an(fld)} researcher",
            {"field": fld, "born": rng.choice(cities).id, "employer": rng.choice(orgs).id},
        )
        people.append(e)
    for e in orgs + people:
        E[e.id] = e
    for o in orgs:
        o.attrs["founder"] = rng.choice(people).id
        if rng.random() < 0.35:
            o.attrs["acquired_by"] = rng.choice([x for x in orgs if x.id != o.id]).id

    triples = set()
    for p in people:
        triples.add((p.id, "born_in", p.attrs["born"]))
        triples.add((p.id, "works_at", p.attrs["employer"]))
    for o in orgs:
        triples.add((o.id, "hq_in", o.attrs["hq"]))
        triples.add((o.id, "founded_by", o.attrs["founder"]))
        if "acquired_by" in o.attrs:
            triples.add((o.id, "acquired_by", o.attrs["acquired_by"]))

    def p_short(p: Entity) -> str:
        return rng.choice(p.aliases[1:])

    def o_any(o: Entity) -> str:
        return rng.choices(o.aliases, weights=[0.5, 0.25, 0.25])[0]

    docs: list[Doc] = []

    for p in people:
        d = _DocBuilder(len(docs), p.id)
        first = p.name if rng.random() < 0.8 else rng.choice(p.aliases[2:])
        d.ent(first, p.id).text(f" is {an(p.attrs['field'])} researcher. ")
        c, o = E[p.attrs["born"]], E[p.attrs["employer"]]
        if rng.random() < 0.5:
            d.ent(p_short(p), p.id).text(" was born in ").ent(c.name, c.id).text(". ")
        else:
            d.ent(c.name, c.id).text(" is the birthplace of ").ent(p_short(p), p.id).text(". ")
        r = rng.random()
        if r < 0.4:
            d.ent(p_short(p), p.id).text(" works at ").ent(o_any(o), o.id).text(".")
        elif r < 0.7:
            d.ent(p_short(p), p.id).text(" joined ").ent(o_any(o), o.id)
            d.text(f" in {rng.randint(2005, 2024)}.")
        else:
            d.ent(o_any(o), o.id).text(" employs ").ent(p_short(p), p.id).text(".")
        d.facts += [(p.id, "born_in", c.id), (p.id, "works_at", o.id)]
        docs.append(d.build())

    for o in orgs:
        d = _DocBuilder(len(docs), o.id)
        d.ent(o.name, o.id).text(f" is {an(o.attrs['sector'])} company. ")
        c, f = E[o.attrs["hq"]], E[o.attrs["founder"]]
        verb = rng.choice([" is headquartered in ", " is based in "])
        d.ent(o_any(o), o.id).text(verb).ent(c.name, c.id).text(". ")
        fm = rng.choices([f.name, f.aliases[2], f.aliases[3]], weights=[0.5, 0.25, 0.25])[0]
        r = rng.random()
        if r < 0.4:
            d.text("It was founded by ").ent(fm, f.id).text(".")
        elif r < 0.7:
            d.ent(fm, f.id).text(f", {an(f.attrs['field'])} researcher, founded ")
            d.ent(o_any(o), o.id).text(".")
        else:
            d.ent(fm, f.id).text(" founded ").ent(o_any(o), o.id).text(".")
        d.facts += [(o.id, "hq_in", c.id), (o.id, "founded_by", f.id)]
        docs.append(d.build())

    for o in orgs:
        if "acquired_by" not in o.attrs:
            continue
        a = E[o.attrs["acquired_by"]]
        d = _DocBuilder(len(docs), None)
        d.ent(o_any(a), a.id).text(" acquired ")
        if rng.random() < 0.5:
            d.text(f"the {o.attrs['sector']} firm ")
        d.ent(o_any(o), o.id).text(f" in {rng.randint(2010, 2024)}.")
        d.facts.append((o.id, "acquired_by", a.id))
        docs.append(d.build())

    for _ in range(n_distractors):
        d = _DocBuilder(len(docs), None)
        c = rng.choice(cities)
        if rng.random() < 0.5:
            p = rng.choice(people)
            d.ent(rng.choice(p.aliases), p.id).text(" gave a talk in ").ent(c.name, c.id)
            d.text(f" about {p.attrs['field']}.")
        else:
            o = rng.choice(orgs)
            d.ent(o_any(o), o.id).text(" opened a sales office in ").ent(c.name, c.id).text(".")
        docs.append(d.build())
    rng.shuffle(docs)
    for i, d in enumerate(docs):
        d.id = i

    questions = _make_questions(E, people, orgs, rng)
    return World(E, docs, questions, triples)


def _person_ref(p: Entity, E: dict[str, Entity]) -> str:
    homonym = sum(e.kind == "person" and e.name == p.name for e in E.values()) > 1
    return f"the {p.attrs['field']} researcher {p.name}" if homonym else p.name


def _make_questions(E, people, orgs, rng, per_type: int = 80) -> list[Question]:
    rel = {
        "born_in": lambda e: e.attrs.get("born"),
        "works_at": lambda e: e.attrs.get("employer"),
        "hq_in": lambda e: e.attrs.get("hq"),
        "founded_by": lambda e: e.attrs.get("founder"),
        "acquired_by": lambda e: e.attrs.get("acquired_by"),
    }
    templates = [
        (people, ["born_in"], "Where was {s} born?"),
        (orgs, ["hq_in"], "In which city is {s} headquartered?"),
        (people, ["works_at", "hq_in"], "In which city is the employer of {s} headquartered?"),
        (orgs, ["founded_by", "born_in"], "Where was the founder of {s} born?"),
        (
            people,
            ["works_at", "founded_by", "born_in"],
            "Where was the founder of the company that employs {s} born?",
        ),
        (
            people,
            ["works_at", "acquired_by", "hq_in"],
            "In which city is the company that acquired the employer of {s} headquartered?",
        ),
    ]
    qs: list[Question] = []
    for pool, path, tpl in templates:
        cands = pool[:]
        rng.shuffle(cands)
        n = 0
        for s in cands:
            chain, support, cur = [], [], s.id
            for r in path:
                nxt = rel[r](E[cur])
                if nxt is None:
                    break
                support.append((cur, r, nxt))
                chain.append(nxt)
                cur = nxt
            if len(chain) != len(path):
                continue
            ref = _person_ref(s, E) if s.kind == "person" else s.name
            qs.append(Question(tpl.format(s=ref), len(path), path, s.id, chain, support))
            n += 1
            if n == per_type:
                break
    return qs
