import json
import urllib.parse
import urllib.request

url = "http://localhost:3030/flights/sparql"
query = """
SELECT DISTINCT ?orig_iata ?dest_iata WHERE {
  ?flight <http://www.semanticweb.org/ontologies/flight_ontology#hasAirportDetails> ?airport .
  OPTIONAL { ?airport <http://www.semanticweb.org/ontologies/flight_ontology#orig_iata> ?orig_iata . }
  OPTIONAL { ?airport <http://www.semanticweb.org/ontologies/flight_ontology#dest_iata> ?dest_iata . }
}
"""
data = urllib.parse.urlencode({
    "query": query,
    "format": "application/sparql-results+json"
}).encode()
req = urllib.request.Request(url, data=data)
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read())
    for b in result["results"]["bindings"]:
        print(b)