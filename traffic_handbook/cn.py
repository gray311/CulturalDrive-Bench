import os
import json

china_traffic_handbook = """S1 Driving side & lane discipline: In mainland China you drive on the right. As a default, vehicles keep to the right unless road markings/signs say otherwise; move left to overtake or to position for a left turn, then return when safe. Where the road is divided into motor-vehicle lanes, non-motor-vehicle lanes, and sidewalks, each category must use its designated part of the road.

S2 Speed limits are in km/h (not mph). Always follow posted limit signs and lane-specific markings. On expressways, passenger vehicles commonly operate up to 120 km/h where permitted; urban roads usually have lower signed limits depending on road class, geometry, and local conditions.

S3 Traffic lights (vehicles): Standard signals are red, yellow, and green. Red means stop, green means go, and yellow means stop if it is safe to do so; vehicles that have already crossed the stop line may continue. Flashing yellow means proceed with caution after checking for safety. If a traffic police officer is directing traffic, the officer’s instructions override the lights.

S4 Right turn on red: In mainland China, a vehicle may generally turn right on red unless a specific red arrow, sign, marking, or lane-control signal prohibits that movement. Even when right turn on red is allowed, the driver must yield to pedestrians and other lawfully moving traffic and must not force through the turn.

S5 Signalised intersections & turning priority: At signalised intersections, enter the correct guide lane early for your intended movement. Turning vehicles must not obstruct released straight-through traffic or pedestrians. Where opposing movements conflict, right-turning vehicles typically yield to opposing left-turning vehicles already lawfully proceeding.

S6 Unsignalised intersections: At intersections without signals or police control, first slow down and observe. If signs or markings are present, follow them. If there is no priority control, yield to traffic approaching from your right; turning vehicles must also yield to straight-through traffic already proceeding through the intersection.

S7 Roundabouts: At roundabouts, vehicles entering must give way to vehicles already circulating within the roundabout. Choose the correct lane early according to your intended exit and do not force entry into the circulating flow.

S8 Pedestrian crossings: A motor vehicle approaching a crosswalk must slow down. If pedestrians are crossing, the vehicle must stop and yield. At intersections, turning vehicles must also yield to pedestrians who are lawfully crossing the road into which the vehicle is turning.

S9 Non-motor vehicle lanes: Many urban roads in China have dedicated lanes for bicycles and electric bicycles. Motor vehicles must not drive in, stop in, or occupy these lanes unless markings explicitly permit it. Drivers must anticipate frequent interaction with bicycles, e-bikes, and scooters near junctions and curbside areas.

S10 Electric bicycles & mixed traffic: Electric bicycles are a major part of mainland China’s road environment and are typically treated under non-motor-vehicle rules when compliant with the legal standard. They commonly use non-motor-vehicle lanes and may appear in large numbers at junctions, crossings, and roadside access points; drivers should expect dense mixed traffic and yield where required.

S11 Yellow box junctions / keep-clear areas: A yellow criss-cross box marking means do not enter unless your exit is clear. Do not stop and wait inside the box, and do not enter an occupied intersection in a way that blocks cross traffic. This rule is especially important in dense urban traffic.

S12 Queueing, merging & lane changes: When traffic is queued or moving slowly, drivers must not force lane changes, cut into queues aggressively, or use opposing lanes to bypass congestion. A vehicle changing lanes must not affect the normal movement of vehicles already in the target lane.

S13 Dedicated lanes & special-use lanes: Roads may include lanes reserved for buses, non-motor vehicles, turning movements, or other special uses. Only the permitted class of traffic may use such lanes during the signed times and conditions. Always check lane arrows, overhead signals, roadside signs, and pavement text.

S14 Expressways / motorways: Expressways in mainland China are controlled-access roads. Use ramps to enter and exit, yield when merging, and do not reverse, make U-turns, or drive the wrong way. The emergency lane must not be used except for emergencies or as specifically directed by authorities.

S15 Emergency vehicles & enforcement: Police cars, fire engines, ambulances, and other emergency vehicles on urgent duty have priority and may use sirens/lights to request immediate passage; other road users must yield. Traffic enforcement in mainland China also relies heavily on cameras and automated monitoring, so violations such as red-light running, speeding, and illegal lane use are commonly detected electronically."""
