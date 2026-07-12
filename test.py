import json
import urllib.parse
import urllib.request

QUERIES = {

    "kg1_core": ("http://localhost:3030/flights/sparql", """
PREFIX fo: <http://www.semanticweb.org/ontologies/flight_ontology#>
SELECT ?number ?gate ?terminal ?callsign ?gspeed ?vspeed WHERE {
  ?flight a fo:Flight ;
          fo:flightNumber ?number .
  OPTIONAL { ?flight fo:hasGate ?gate }
  OPTIONAL { ?flight fo:hasTerminal ?terminal }
  OPTIONAL { ?flight fo:hasCallsign ?callsign }
  OPTIONAL { ?flight fo:hasFlightEvent ?event . ?event fo:gspeed ?gspeed ; fo:vspeed ?vspeed }
} LIMIT 15
"""),

    "kg1_airline_route": ("http://localhost:3030/flights/sparql", """
PREFIX fo: <http://www.semanticweb.org/ontologies/flight_ontology#>
SELECT ?number ?airlineCode ?origCity ?destCity WHERE {
  ?flight a fo:Flight ;
          fo:flightNumber ?number .
  OPTIONAL { ?flight fo:hasAirline ?al . ?al fo:operating_as ?airlineCode }
  OPTIONAL { ?flight fo:hasOriginCity ?oc . ?oc fo:orig_city ?origCity }
  OPTIONAL { ?flight fo:hasDestinationCity ?dc . ?dc fo:dest_city ?destCity }
} LIMIT 15
"""),

    "kg1_dest_iata": ("http://localhost:3030/flights/sparql", """
PREFIX fo: <http://www.semanticweb.org/ontologies/flight_ontology#>
SELECT ?number ?destIata WHERE {
  ?flight a fo:Flight ;
          fo:flightNumber ?number ;
          fo:hasAirportDetails ?ad .
  ?ad fo:dest_iata ?destIata .
} LIMIT 15
"""),

    "kg2_core": ("http://localhost:3030/airports/sparql", """
PREFIX ao: <http://www.semanticweb.org/ontologies/airport_ontology#>
SELECT ?iata ?name ?elevation ?type ?municipality WHERE {
  ?airport a ao:Airport ;
           ao:iataCode ?iata ;
           ao:airportName ?name .
  OPTIONAL { ?airport ao:elevationFt ?elevation }
  OPTIONAL { ?airport ao:airportType ?type }
  OPTIONAL { ?airport ao:municipality ?municipality }
} LIMIT 15
"""),

    "kg2_country": ("http://localhost:3030/airports/sparql", """
PREFIX ao: <http://www.semanticweb.org/ontologies/airport_ontology#>
SELECT ?iata ?countryName WHERE {
  ?airport a ao:Airport ;
           ao:iataCode ?iata ;
           ao:locatedInCountry ?c .
  ?c ao:countryName ?countryName .
} LIMIT 15
"""),

    "kg2_runway": ("http://localhost:3030/airports/sparql", """
PREFIX ao: <http://www.semanticweb.org/ontologies/airport_ontology#>
SELECT ?iata ?length ?surface WHERE {
  ?airport a ao:Airport ;
           ao:iataCode ?iata ;
           ao:hasRunway ?rw .
  ?rw ao:lengthFt ?length ; ao:surface ?surface .
} LIMIT 15
"""),

    "kg3_professors": ("http://localhost:3030/university/sparql", """
PREFIX ub: <http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#>
SELECT ?person ?name ?deptName WHERE {
  ?person a ?type ;
          ub:name ?name ;
          ub:worksFor ?dept .
  ?dept ub:name ?deptName .
  FILTER(CONTAINS(STR(?type), "Professor") || CONTAINS(STR(?type), "Lecturer"))
} LIMIT 15
"""),

    "kg3_courses": ("http://localhost:3030/university/sparql", """
PREFIX ub: <http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#>
SELECT ?personName (COUNT(?course) AS ?numCourses) WHERE {
  ?person ub:name ?personName ;
          ub:teacherOf ?course .
} GROUP BY ?personName ORDER BY DESC(?numCourses) LIMIT 15
"""),

    "kg3_students": ("http://localhost:3030/university/sparql", """
PREFIX ub: <http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#>
SELECT ?name ?deptName WHERE {
  ?student a ub:GraduateStudent ;
           ub:name ?name ;
           ub:memberOf ?dept .
  ?dept ub:name ?deptName .
} LIMIT 15
"""),

}

results = {}
for name, (endpoint, q) in QUERIES.items():
    try:
        data = urllib.parse.urlencode({
            "query": q,
            "format": "application/sparql-results+json"
        }).encode()
        with urllib.request.urlopen(urllib.request.Request(endpoint, data=data)) as r:
            results[name] = json.loads(r.read())["results"]["bindings"]
        print(f"[ok] {name}: {len(results[name])} rows")
    except Exception as e:
        print(f"[FAILED] {name}: {e}")
        results[name] = []

with open("kg_sample_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nSaved to kg_sample_data.json")