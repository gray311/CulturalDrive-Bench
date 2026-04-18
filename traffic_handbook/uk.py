import os
import json


uk_traffic_handbook = """S1 Driving side & lane discipline: In the UK you drive on the left. As a default, keep left unless road markings/signs say otherwise; move out to overtake, to turn right, or to pass an obstruction, then return to the left when safe.

S2 Speed limits are in mph (not km/h). Unless signed otherwise, typical “national speed limit” defaults are 70 mph on motorways/dual carriageways and 60 mph on single carriageways, while built-up areas are commonly 30 mph (and widely 20 mph zones exist in many places). Always follow posted limit signs and the national speed limit symbol where used.

S3 Traffic lights (vehicles): Standard UK sequence is red → red+amber (get ready, still stop) → green → amber (stop unless unsafe). There is no general “turn on red” permission—treat a red signal as stop unless a separate permitted movement is explicitly shown.

S4 Green filter arrows at signals: A green arrow indicates a permitted movement for a specific lane/direction (a “filter”). Only enter that lane if you intend to go that way, and proceed only in the arrow’s direction when it is lit (even if other movements are held).

S5 Roundabouts: Modern UK-style roundabouts use the “priority rule”: traffic entering must give way to traffic already circulating. In left-hand traffic this generally means giving way to vehicles approaching from your right; choose the correct lane early and signal on exit.

S6 “Give Way” / “Stop” control: At many junctions you’ll see a GIVE WAY sign and/or an inverted triangle road marking with broken white lines—yield to traffic on the main road. A STOP sign and solid stop line require a full stop before proceeding.

S7 Zebra/parallel crossings (uncontrolled): Zebra crossings are marked with black-and-white stripes and Belisha beacons. Drivers should be prepared to slow/stop for pedestrians waiting, and must give way once a pedestrian has stepped onto the crossing. Do not overtake the vehicle nearest a crossing (especially if it has stopped).

S8 Signal-controlled pedestrian crossings: Pelican, puffin, toucan and related crossings use red/green “man” signals for pedestrians with push-buttons. Pelican crossings are distinctive because after the vehicle red phase, a flashing amber may allow vehicles to proceed if the crossing is clear; puffin/toucan crossings do not use a flashing amber phase and instead follow a standard traffic-light style sequence.

S9 Box junctions (“yellow boxes”): A criss-cross yellow box means “don’t enter unless your exit is clear.” Exception: you may enter and wait if you are turning right and are only blocked by oncoming traffic or other right-turning vehicles; at signalled roundabouts, do not enter the box unless you can fully clear it without stopping.

S10 Yellow-line parking/waiting rules: Double yellow lines mean no waiting at any time (subject to signed exceptions). Single yellow lines restrict waiting during times shown on nearby plates or zone-entry signs. Short yellow kerb marks (“blips”) add loading/unloading restrictions—check signs.

S11 Red routes (where used): Red lines along the kerb indicate stronger “no stopping” controls than yellow lines. Double red lines mean no stopping at any time; single red lines apply during signed times. Only stop in specifically marked bays/boxes when permitted by accompanying signs.

S12 Motorways basics: Motorways are controlled-access roads—no pedestrians/cyclists and no at-grade junctions; you join via a slip road and must give priority to motorway traffic. Use the left lane as your default cruising lane and overtake to the right; do not weave.

S13 Hard shoulder / emergency areas / overhead control: Do not use a hard shoulder except in an emergency or when directed by signs/authorities. On some motorways the hard shoulder may be opened as a running lane only when overhead signs show it is open; a red ‘X’ indicates a closed lane that must not be used. Use emergency areas only for emergencies and follow SOS signage.

S14 Bus lanes & bus gates: Many UK cities use bus lanes (and “bus gates”) reserved for buses (often also cycles and sometimes taxis) during signed hours; entering when not permitted is an offence. Look for carriageway text (e.g., “BUS LANE” / “BUS GATE”) and regulatory signs.

S15 Vulnerable road users & junction priority: Expect frequent interaction with pedestrians, cyclists, and (in some areas) horses. Give extra space when overtaking vulnerable road users; at junctions, give way to pedestrians crossing or waiting to cross the road you are turning into/from, and respect cyclist advanced stop lines at signals (stop behind the first line on red/amber)."""