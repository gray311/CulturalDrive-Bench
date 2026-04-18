import os
import json

singapore_traffic_handbook = """S1 Driving side & lane discipline: In Singapore you drive on the left. As a default, keep left on two-way roads and dual carriageways unless road markings/signs say otherwise; move right to overtake, to turn right, or to avoid an obstruction, then return left when safe. On roads with two lanes, the left lane is for normal driving and the right lane is mainly for overtaking and right turns; on roads with three lanes, the left lane is for slower vehicles, the centre lane for faster vehicles, and the outer right lane for overtaking and right turns.

S2 Speed limits are in km/h (not mph). Unless otherwise stated, the speed limit on roads in Singapore is generally 50 km/h. Always follow posted limit signs and special reduced-speed zones such as School Zones and Silver Zones.

S3 Traffic lights (vehicles): Standard signals are green, amber, and red. Green means proceed only if the way ahead is clear and the vehicle can fully clear the junction. Drivers must not enter a junction if doing so would cause obstruction (e.g., yellow-box junction). Amber means stop unless too close to stop safely. Red means stop behind the stop line. Even with a green light, drivers must watch for pedestrians, vehicles still clearing the junction, and unexpected hazards.

S4 Left Turn on Red (LTOR): There is no general “turn on red” rule. At junctions with a specific Left Turn on Red sign, drivers may turn left only after making a full stop, giving way to pedestrians, and giving way to traffic approaching from the right before proceeding when safe.

S5 Right-turn control & arrows: Many junctions use dedicated right-turn signals (green arrow). When a red arrow is shown, vehicles must not turn. Right-turning vehicles must give way to oncoming traffic going straight and to pedestrians crossing. Right turns are only permitted when signals or conditions explicitly allow and the path is clear. :contentReference[oaicite:0]{index=0}

S6 Junction priority rules:
- At signalised junctions:
  Follow traffic lights. Vehicles going straight generally have priority over turning vehicles. Turning vehicles must give way to oncoming traffic and pedestrians.
- At unsignalised or uncontrolled junctions:
  Drivers must give way to vehicles approaching from the right when no signals or priority signs are present. :contentReference[oaicite:1]{index=1}
- At major/minor road junctions:
  Vehicles entering from a minor road must give way to traffic on the major road.

S7 Roundabouts: Slow down when approaching a roundabout and give way to traffic already on the roundabout, typically approaching from the right. Enter only when safe and do not block exits.

S8 Yellow-box junctions: Do not enter the yellow box unless the exit road is clear and you can fully clear the junction. This applies even if the traffic light is green.

S9 Pedestrian priority: Drivers must slow down and be prepared to stop for pedestrians at crossings. When turning left or right at a junction, drivers must give way to pedestrians crossing the road into which they are turning.

S10 Bus lanes: Bus lanes operate during specified hours. Other vehicles must not use them during restricted times. Always check roadside signs.

S11 Bus Priority Box / Give Way to Buses: Drivers must give way to buses exiting bus stops where Bus Priority markings are present. Stop before the give-way line and do not block the box.

S12 School Zones & Silver Zones: These are low-speed safety zones. Drivers must reduce speed and watch carefully for vulnerable road users such as children and elderly pedestrians.

S13 Expressways & tunnels: Controlled-access roads where pedestrians are not allowed. Use designated entry/exit ramps and follow lane-use signals. Do not stop except in emergencies.

S14 Lane-use signals & overhead signs: A red “X” indicates a closed lane. Drivers must obey all overhead signals and variable message signs.

S15 Distinctive Singapore road features & signs: Singapore roads commonly include regulatory blue circular signs, GIVE WAY and STOP markings, LTOR signs, right-turn arrows, ERP signs, and bus priority markings. Drivers must always follow specific signs and road markings over general rules."""