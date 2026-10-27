-- BeamNG.Earth: the line-of-sight horizon (docs/V1_4_SPEC.md 3.6 "Radius", milestone M1h; the
-- server's --horizon, announced by hello.horizon). The server's far field then reaches the real
-- horizon (status.radius, 80 km to 1500 km), and core.lua uses this module to make the scene match:
--   * LevelInfo.visibleDistance follows radius * VISIBLE_MARGIN, never below the mod's 100 km and
--     with VISIBLE_HYSTERESIS; every change is followed by postApply(): in-game (2026-10-10,
--     docs/research/view_distance/) assigning the field alone does nothing, the engine keeps the
--     level file's value until postApply.
--   * ScatterSky.atmoPlanetRadius = PLANET_RADIUS_KM. The sky is a physical atmosphere around a
--     sphere of that radius with the camera at camPos.z above it (our z ~ altitude), so its horizon
--     already dips with height; the level has 6360 km, our tiles are curved with 6371 km (the server's
--     farfield.R_EARTH), so both horizons use one sphere. The level's value comes back when horizon
--     mode ends (a server without it) or before the level unloads.
--
-- Distance haze (the M1h TODO) is done, in haze.lua: the settings' hazeStrength drives
-- core_environment.setFogDensity / setFogAtmosphereHeight, and the fade now reaches high enough to
-- cover what the far field draws (the level's fogAtmosphereHeight of 1000 m was the hard edge, not
-- its density). Still open there: the sky's own aerial perspective, ScatterSky atmoMieScattering /
-- atmoMieExtinction (setField(name, 0, 'r g b'), as celestial.lua applyAtmosphere does) and
-- atmoMieScaleHeight, which is the physical way to colour the haze with the sun's position.

local M = {}

M.PLANET_RADIUS_KM = 6371     -- the tiles' sphere (server farfield.R_EARTH / 1000)
M.VISIBLE_MARGIN = 1.1        -- visibleDistance = radius + 10 % (server farfield.VISIBLE_MARGIN)
M.VISIBLE_HYSTERESIS = 0.1    -- re-set only when the target moved by more than 10 % of the current value

-- Pure: the visibleDistance (m) to set for a far-field radius (m), given the current value and the
-- minimum; nil to keep the current one (within the hysteresis, or no usable radius).
function M.visibleTarget(radius, current, minimum)
  radius = tonumber(radius)
  if not radius or radius ~= radius or radius <= 0 or radius == math.huge then return nil end
  local want = math.max(tonumber(minimum) or 0, radius * M.VISIBLE_MARGIN)
  current = tonumber(current)
  if current and current > 0 and math.abs(want - current) <= M.VISIBLE_HYSTERESIS * current then return nil end
  return want
end

-- LevelInfo.visibleDistance = v, then postApply (without it the engine keeps the level's value).
function M.setVisibleDistance(v)
  local li = scenetree.theLevelInfo
  if not li then return false end
  li.visibleDistance = v
  if li.postApply then li:postApply() end
  return true
end

local saved = nil -- { id, radius }: the sky object and its atmoPlanetRadius before applySky

local function sky()
  local names = scenetree.findClassObjects('ScatterSky')
  return names and names[1] and scenetree.findObject(names[1]) or nil
end

-- The sky's planet radius = the tiles' (remembering the level's value once per sky object). Runs on
-- every status message in horizon mode, so the field is written only when it differs.
function M.applySky()
  local s = sky()
  if not s then return false end
  if not saved or saved.id ~= s:getID() then saved = { id = s:getID(), radius = tonumber(s.atmoPlanetRadius) } end
  if tonumber(s.atmoPlanetRadius) ~= M.PLANET_RADIUS_KM then s.atmoPlanetRadius = M.PLANET_RADIUS_KM end
  return true
end

-- Back to the level's planet radius (if applySky changed it and the sky still exists).
function M.restoreSky()
  if not saved then return false end
  local s = scenetree.findObjectById(saved.id)
  if s and saved.radius then s.atmoPlanetRadius = saved.radius end
  saved = nil
  return s ~= nil
end

function M.skyApplied() return saved ~= nil end

return M
