import os
import json

us_traffic_handbook = """S1 Driving side & lane discipline: In the United States you drive on the right. As a default, keep right on two-way roads unless markings/signs say otherwise; move left to overtake, to position for a left turn, or to avoid an obstruction, then return to the right when safe.

S2 Speed limits are in mph (not km/h). There is no single nationwide default speed limit for all roads; limits vary by state, road type, and local conditions. Always follow posted limit signs, and expect lower signed speeds in school zones, work zones, residential streets, and other special areas.

S3 Traffic lights (vehicles): Standard signals are green, yellow, and red. Green permits movement subject to yielding where required; yellow means stop if you can do so safely; red means stop at the stop line, before the crosswalk, or before entering the intersection. A flashing red signal is treated like a STOP sign, while a flashing yellow signal means proceed with caution.

S4 Turn on red: A major U.S. feature is that right turn on red is generally permitted after a full stop unless a sign prohibits it, but this varies by state and locality. Even where allowed, the driver must yield to pedestrians, cyclists, and other traffic with the right-of-way. Do not assume that a red arrow allows the turn; red-arrow rules vary and are often more restrictive.

S5 All-way stop / four-way stop: All-way stop intersections are especially common in the U.S. Every approaching vehicle must come to a full stop. The first vehicle to arrive generally goes first; if two vehicles arrive at the same time, the vehicle on the right typically has priority. When opposite vehicles arrive together, a left-turning vehicle normally yields to an oncoming vehicle going straight.

S6 STOP / YIELD control: At a STOP sign, stop at the stop line; if there is no stop line, stop before the crosswalk or before entering the intersection where you can see safely. At a YIELD sign, slow down and give way to conflicting traffic and pedestrians before proceeding.

S7 Pedestrian crossings & turning priority: Drivers must yield to pedestrians in crosswalks. At many intersections, this includes marked crosswalks and also unmarked crosswalks implied by the intersection geometry under state law. Turning vehicles must not cut across pedestrians who are lawfully crossing.

S8 School buses: School-bus rules are a major U.S.-specific feature. Yellow flashing lights mean the bus is preparing to stop, so slow down and prepare to stop. Red flashing lights with an extended stop arm mean you must stop and remain stopped until the lights stop flashing, the stop arm is withdrawn, and the bus begins moving. On undivided roads, traffic in both directions generally must stop; on divided highways, opposing traffic often does not need to stop, depending on state law and roadway separation.

S9 Freeways / Interstates: U.S. freeways and Interstates are controlled-access roads. Use ramps to enter and exit, accelerate on the entrance ramp, and yield to freeway traffic when merging. Use the right lanes for ordinary travel and the left lanes mainly for overtaking, unless local signs or lane controls specify otherwise.

S10 HOV / carpool / managed lanes: Many U.S. urban highways use HOV or managed lanes reserved for vehicles meeting posted occupancy or toll requirements. These lanes are commonly marked by a white diamond symbol and/or HOV wording. Always check the posted occupancy requirement, hours of operation, and entry/exit restrictions.

S11 Lane-control signals / reversible lanes: Some roads use overhead lane-control signals. A green downward arrow means the lane is open for travel; a red X means the lane is closed and must not be used. On roads with reversible or actively managed lanes, follow overhead signals and signs rather than assuming the lane direction from memory.

S12 Two-way center left-turn lanes: A common U.S. road feature is the shared center left-turn lane on multi-lane roads. Use it only for left turns or, where permitted, short setup movements into driveways or side streets. Do not use it as a through lane, passing lane, or waiting lane for general travel.

S13 Railroad crossings: At railroad crossings, obey flashing lights, bells, gates, signs, and pavement markings. Stop behind the stop line or before the tracks when required, and do not enter unless you can clear the tracks completely. Never drive around or under a lowered gate.

S14 Emergency vehicles & roadside incidents: When an emergency vehicle approaches with siren/lights, pull to the right and stop unless directed otherwise. Many states also have “move over” or slow-down laws for stopped emergency, service, or disabled vehicles on the roadside, so check local law and follow posted requirements.

S15 Sign system note: U.S. road signs are largely standardized through the MUTCD. For visual recognition, common clues include the octagonal STOP sign, the triangular YIELD sign, yellow diamond warning signs, school-zone signs, white regulatory speed-limit signs in mph, and white diamond markings/signs for HOV or carpool lanes."""
