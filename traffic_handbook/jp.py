import os
import json

japan_traffic_handbook = """S1 Driving side & lane discipline: In Japan you drive on the left. As a default, keep left unless road markings/signs say otherwise; move out to overtake, to turn right, or to pass an obstruction, then return to the left when safe.

S2 Speed limits are in km/h (not mph). Unless signed otherwise, regular-sized automobiles must obey default speed limits of 60 km/h on ordinary roads and 100 km/h on expressways. Always follow posted limit signs and road markings where they are provided.

S3 Traffic lights (vehicles): Standard signals are green, yellow, and red. Green permits through movement and turning for ordinary motor vehicles. Yellow means stop unless you are already too close to stop safely. Red means stop at the stopping point.

S4 Turn on red / arrow signals: There is no general “turn on red” permission in Japan. A vehicle must not proceed past the stopping point on red unless it is already in the process of completing a lawful turn through the intersection. A green arrow permits movement in the indicated direction even if the main signal is red or yellow.

S5 Left-turn and right-turn behaviour at signals: In left-hand traffic, left turns are usually the simpler near-side turn, while right turns typically cross opposing traffic and may require waiting within the intersection for a safe gap. A vehicle already making a right turn must not obstruct traffic approaching on a green light.

S6 Two-stage right turn for light vehicles / mopeds: Lightweight vehicles such as bicycles, and mopeds where two-stage right turn rules apply, do not make a direct sweeping right turn like a car. Instead, they proceed straight to the far side, stop at the turning point, reorient, and wait for the next release to complete the right turn.

S7 “Stop” / “Slow down” control: At a STOP sign or stop line, make a full stop before proceeding. Japan’s STOP sign is visually distinctive: an inverted red triangle rather than the octagonal design common in many other countries. “Slow down” control also appears as a specific sign and requires reduced speed and caution.

S8 Railway crossings: A major Japan-specific rule is that drivers must stop before a railway crossing (or before the stop line if there is one), check both directions, and proceed only when safe. If the crossing is controlled by traffic lights, pass in accordance with the lights. Do not enter unless there is space to clear the tracks fully.

S9 Pedestrian crossings: Pedestrians have right-of-way on pedestrian crossings. Drivers must slow down when approaching a pedestrian crossing or bicycle crossing lane unless it is clearly empty, and must stop and yield when pedestrians or cyclists are crossing or about to cross.

S10 No overtaking near crossings: Do not overtake a vehicle that is stopped at or immediately before a pedestrian crossing or bicycle crossing lane. Drivers must also not overtake another vehicle and then cut in across a crossing, or within 30 meters before such a crossing.

S11 Parking / stopping near crossings: Drivers must not stop or park on a pedestrian crossing or bicycle crossing lane, or within the restricted area around it, except when stopping for a red light or to avoid danger. Watch carefully for posted “No parking” and “No stopping” signs.

S12 Expressways (motorways): Expressways are controlled-access roads. Use the acceleration lane to build speed before merging, yield appropriately when joining, and use the passing lane only for overtaking. Do not drive on the shoulder except in emergencies.

S13 Lane use on expressways: On expressways, stay in the normal driving lane as your default lane and use the passing lane only when overtaking. Do not cruise continuously in the passing lane, and do not weave between lanes unnecessarily.

S14 Vulnerable road users: Expect frequent interaction with pedestrians and cyclists, especially in urban areas and around crossings. Drivers must yield at pedestrian crossings and should watch carefully when turning at intersections, because pedestrians and bicycles may be crossing even when motor traffic also has a green signal.

S15 Sign system note: Road signs in Japan are standardized nationally. For visual recognition tasks, two especially distinctive features are the inverted triangular STOP sign and the prominence of dedicated signs for controls such as “slow down,” “no parking,” “no stopping,” railway crossings, and exclusive lanes."""
