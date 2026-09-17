"""
Sailing-yacht routing: polar performance model, boat environment and planning baselines.

Modules
    polar      boat speed = f(true wind speed, true wind angle), synthetic or loaded from a .pol table
    boat_env   Gymnasium environment: heading control, tack/gybe time penalty
    isochrone  isochrone router (the method commercial weather routers use) -- the SPEED reference
    dp_sail    value iteration on (x, y, heading) -- the OPTIMALITY reference

Conventions (shared by every module)
    positions   domain units; 1 unit = SailParams.nm_per_unit nautical miles
    time        hours
    headings    radians, maths convention: 0 = +x (east), counter-clockwise positive
    wind        a `wind.WindField` giving the direction the air moves TOWARDS, in wind units;
                SailParams.kts_per_wind_unit converts it to knots for the polar
"""
