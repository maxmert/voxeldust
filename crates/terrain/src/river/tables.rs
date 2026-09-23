//! ★ THE WIDTH LAW'S TWO COMMITTED TABLES (slice 8d, ruling B2 step 4; ruling W4 item 1).
//!
//! The hydraulic geometry of **Leopold & Maddock 1953** (USGS Professional Paper 252), calibrated
//! on EARTH and at EARTH'S GRAVITY: `w = 3.9·√Q` metres and `d = 0.4·Q^0.4` metres, with the
//! downstream exponents `b = 0.5` and `f = 0.4` they measured over the Great Plains and the
//! south-west. The two coefficients are CALIBRATIONS and are named as such; the gravity factor is
//! not a calibration and is applied at the read ([`super::channel_width_mm`]).
//!
//! **WHY A TABLE.** The float fence forbids a power, and the row already carries the discharge as a
//! LOG CLASS — `4·log₂(D + 1)` over `D` in mm·m²/yr — so 256 committed entries indexed by that
//! class ARE the law, exact on every host. Each entry stands at its own class's MIDPOINT, so the
//! quantisation costs at most half a class step: 2.5 % on a width, 2 % on a depth.
//!
//! **THE UNIT.** `D` is the rain accumulated over the area, in mm·m²/yr; the discharge in m³/s is
//! `D / 1000 / 31 557 600`, the Julian year — the same year the climate's own rain law is
//! calibrated on (`EARTH_MEAN_RAIN_MM_YR = 990` is millimetres in an EARTH year).
//!
//! **THE LAW'S OWN NUMBERS** (the table `03_erosion_rivers.md` §6.1 committed, reproduced here to
//! within the class step):
//!
//! | drainage area | discharge | class | width | depth |
//! |---|---|---|---|---|
//! | 100 km² | 1.0 m³/s | 139 | 3.9 m | 0.40 m |
//! | 1 000 km² | 9.5 m³/s | 152 | 12.2 m | 1.00 m |
//! | 10 000 km² | 95 m³/s | 165 | 38.2 m | 2.48 m |
//! | 100 000 km² | 951 m³/s | 178 | 117.4 m | 6.09 m |
//! | 1 000 000 km² | 9 506 m³/s | 192 | 390.7 m | 15.95 m |
//! | 6 000 000 km² (an Amazon) | 57 034 m³/s | 202 | 939.1 m | 32.16 m |
//!
//! This file is GENERATED and committed; it is never computed at build time, because a `powf` at
//! build time is a float the fence would not see (§12 D4 of the rivers investigation).

/// The channel's width in whole MILLIMETRES at Earth's gravity, by discharge class.
pub const CHANNEL_WIDTH_MM: [u32; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
    6, 7, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 22, 24, 26, 29, 31, 34, 37, 41, 44, 48,
    53, 57, 62, 67, 75, 81, 87, 95, 105, 115, 123, 135, 149, 162, 174, 191, 211, 229, 246, 270,
    298, 324, 348, 382, 422, 459, 493, 540, 596, 648, 697, 763, 844, 917, 985, 1079, 1193, 1297,
    1393, 1526, 1687, 1834, 1970, 2158, 2386, 2594, 2786, 3052, 3374, 3668, 3940, 4316, 4772, 5188,
    5572, 6104, 6748, 7336, 7880, 8633, 9544, 10375, 11145, 12208, 13497, 14673, 15761, 17265,
    19087, 20750, 22289, 24417, 26994, 29345, 31522, 34531, 38175, 41501, 44579, 48834, 53988,
    58691, 63044, 69061, 76350, 83001, 89158, 97667, 107975, 117381, 126088, 138122, 152700,
    166002, 178315, 195334, 215950, 234763, 252176, 276245, 305400, 332005, 356630, 390669, 431901,
    469526, 504351, 552489, 610800, 664009, 713260, 781338, 863801, 939051, 1008703, 1104978,
    1221600, 1328019, 1426521, 1562675, 1727603, 1878102, 2017405, 2209957, 2443199, 2656037,
    2853042, 3125351, 3455205, 3756204, 4034811, 4419913, 4886398, 5312075, 5706084, 6250702,
    6910411, 7512408, 8069621, 8839827, 9772796, 10624150, 11412167, 12501403, 13820821, 15024817,
    16139242, 17679654, 19545593, 21248299, 22824335, 25002806, 27641642, 30049633, 32278484,
    35359308, 39091185, 42496599, 45648670, 50005612, 55283285, 60099267, 64556968, 70718615,
    78182371, 84993198, 91297340,
];

/// The channel's depth in whole MILLIMETRES at Earth's gravity, by discharge class.
pub const CHANNEL_DEPTH_MM: [u32; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
    2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11, 12, 13, 14, 14, 16,
    17, 18, 19, 21, 22, 24, 25, 27, 29, 31, 33, 36, 39, 41, 44, 47, 51, 55, 58, 62, 67, 72, 76, 82,
    89, 95, 101, 108, 118, 126, 133, 143, 155, 166, 176, 189, 205, 219, 232, 249, 270, 289, 306,
    329, 356, 381, 403, 434, 470, 503, 532, 572, 620, 663, 702, 755, 818, 875, 927, 997, 1080,
    1155, 1223, 1315, 1425, 1523, 1613, 1735, 1880, 2010, 2129, 2290, 2481, 2652, 2809, 3021, 3274,
    3500, 3706, 3987, 4320, 4618, 4890, 5260, 5700, 6094, 6453, 6941, 7521, 8041, 8514, 9159, 9924,
    10610, 11235, 12085, 13095, 14000, 14825, 15946, 17279, 18473, 19561, 21041, 22800, 24375,
    25811, 27764, 30084, 32163, 34058, 36635, 39696, 42440, 44940, 48340, 52380, 55999, 59298,
    63785, 69115, 73892, 78245, 84164, 91198, 97501, 103244, 111055, 120337, 128653, 136232,
    146538, 158786, 169759, 179759, 193359, 209519, 223998, 237194, 255138, 276462, 295567, 312979,
    336657, 364794, 390003, 412978, 444221, 481348, 514612, 544928, 586153, 635143, 679034, 719036,
    773434, 838076, 895991, 948774, 1020552, 1105848, 1182267, 1251915,
];
