from router import _has_ask_signal, _detect_flight_number_first

q = "Is BR62's callsign EVA062?"
print("has_ask_signal:", _has_ask_signal(q))
print("flight_entity:", _detect_flight_number_first(q))