-- Distance haze (horizon.lua's TODO(M1h); the user, 2026-10-27: the map's visual quality). Real
-- visibility is haze-limited to 25-150 km on most days, so distant land should fade to a faint
-- silhouette. Ours ended at a hard edge, and the reason is in the level, not in the density: our
-- LevelInfo has fogAtmosphereHeight 1000 (levels/beamng_earth/main/MissionGroup/Sky), so the fog is a
-- layer a kilometre deep over the ground and everything the far field draws above it -- a hill at
-- 60 km, a mountain at 300 km, the curve of the tiles themselves -- comes out of it perfectly crisp.
-- Stock levels never show this: they fog 75-800 m deep and see 5-25 km, so nothing of theirs is ever
-- both far away and high up.
--
-- The settings' hazeStrength (0-1, 0 by default) is read as a visibility, by Koschmieder's law
-- (extinction * meteorological visibility = CONTRAST, the 2 % contrast threshold):
--
--     0      the level's own fog, untouched (nothing is written, so no player's look changes)
--     0.34   VIS_CLEAR^0.66 * VIS_THICK^0.34 = 98 km, the density our level already carries (4e-5),
--            now over HEIGHT_M instead of 1000 m: the same air, the far edge gone
--     1      25 km, a thick summer haze
--
-- HEIGHT_M is 6 km rather than the aerosol layer's real ~1.5 km on purpose: it is not a physical
-- depth here but how high the fade has to reach to cover what we draw, and the far field draws
-- 4000 m peaks and its own curvature. The fog colour is the level's and is left alone (the sky's
-- aerial perspective, ScatterSky atmoMieScattering, is the physical way to colour it and is a
-- separate job: celestial.lua owns those fields and the weather module writes the sky).
local M = {}

M.CONTRAST = 3.912            -- Koschmieder: extinction coefficient * visibility
M.VIS_CLEAR_M = 200000        -- strength 0+: a very clear day
M.VIS_THICK_M = 25000         -- strength 1: a thick haze
M.HEIGHT_M = 6000             -- the fog layer's depth while the haze is on (the level's is 1000)

-- Pure: (fog density, fog atmosphere height in m) for a strength, or nil to leave the level alone.
function M.fogFor(strength)
  local s = tonumber(strength) or 0
  if not (s > 0) then return nil end
  if s > 1 then s = 1 end
  local vis = M.VIS_CLEAR_M * (M.VIS_THICK_M / M.VIS_CLEAR_M) ^ s
  return M.CONTRAST / vis, M.HEIGHT_M
end

-- The meteorological visibility (m) a strength stands for, for the settings page's label.
function M.visibilityFor(strength)
  local d = M.fogFor(strength)
  return d and M.CONTRAST / d or nil
end

local saved = nil             -- { density, height } of the level, before the haze was first applied
local wanted = nil            -- { density, height } last asked for (nil: off)

local function env()
  return core_environment
end

-- Remember the level's own fog once, so restore() can put it back whatever the slider did since.
local function remember()
  if saved then return true end
  local e = env()
  if not e or not e.getState then return false end
  local ok, st = pcall(e.getState)
  if not ok or type(st) ~= 'table' then return false end
  -- getState reports fogDensity x1000 (core/environment.lua), the setter takes it unscaled
  local d = tonumber(st.fogDensity)
  local h = tonumber(st.fogAtmosphereHeight)
  if not d or not h then return false end
  saved = { density = d / 1000, height = h }
  return true
end

-- Write the fog for `strength` (nil / 0: the level's own again). Returns true when the engine took it.
function M.set(strength)
  local e = env()
  if not e or not e.setFogDensity or not e.setFogAtmosphereHeight then return false end
  local d, h = M.fogFor(strength)
  if d == nil then
    wanted = nil
    return M.restore()
  end
  if not remember() then return false end
  wanted = { density = d, height = h }
  e.setFogDensity(d)
  e.setFogAtmosphereHeight(h)
  return true
end

-- Leaving our level, or the haze turned off: the level's own fog again.
function M.restore()
  if not saved then return false end
  local e = env()
  local s = saved
  saved, wanted = nil, nil
  if not e or not e.setFogDensity or not e.setFogAtmosphereHeight then return false end
  e.setFogDensity(s.density)
  e.setFogAtmosphereHeight(s.height)
  return true
end

function M.applied() return saved ~= nil end

-- Every frame in our level, as postfx.update: the fog is re-written when something else reset it
-- (the weather module and the game's own presets write LevelInfo too), checked every CHECK_S.
M.CHECK_S = 2
local lastCheck = -math.huge
function M.update(inOurLevel, strength)
  -- off (another level, or the slider back to 0) is handled before the throttle: it only spares us
  -- reading getState every frame, and it must not hold the haze on for a second after it is turned off
  local d = inOurLevel and M.fogFor(strength) or nil
  if d == nil then
    if saved then M.restore() end
    return
  end
  local t = os.clock()
  if t - lastCheck < M.CHECK_S then return end
  lastCheck = t
  local e = env()
  local st = e and e.getState and select(2, pcall(e.getState)) or nil
  local now = type(st) == 'table' and tonumber(st.fogDensity) or nil
  -- getState's density is x1000; rewrite when it drifted by more than a thousandth of itself
  if not wanted or now == nil or math.abs(now / 1000 - wanted.density) > wanted.density * 1e-3 then
    M.set(strength)
  end
end

return M
