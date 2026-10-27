-- luajit mod/tests/test_haze.lua  (distance haze: horizon.lua's M1h TODO, settings hazeStrength)
local T = dofile((arg[0]:match('^(.*)[/\\]') or '.') .. '/testlib.lua')
local H = require('beamngEarth/haze')

-- strength 0: nothing is written, so no player's look changes by upgrading
T.check(H.fogFor(0) == nil, 'strength 0: the level\'s own fog')
T.check(H.fogFor(nil) == nil and H.fogFor(-1) == nil and H.fogFor('x') == nil, 'no strength: the level\'s own fog')

-- Koschmieder both ways, and the level's own density at the strength that stands for 98 km
local d1, h1 = H.fogFor(1)
T.near(d1, H.CONTRAST / H.VIS_THICK_M, 1e-12, 'strength 1 is the thick-haze visibility')
T.check(h1 == H.HEIGHT_M, 'the fog layer is deepened to cover the far field')
T.near(H.visibilityFor(1), H.VIS_THICK_M, 1.0, 'strength 1 reads back as its visibility')
T.check(H.fogFor(2) == H.fogFor(1), 'strength above 1 is clamped')
local s98 = math.log(H.VIS_CLEAR_M / 97800) / math.log(H.VIS_CLEAR_M / H.VIS_THICK_M)
T.near(select(1, H.fogFor(s98)), 4e-5, 2e-7, 'the level\'s own 4e-5 is ~98 km of visibility')
-- monotone: more haze is never more visibility
local prev = math.huge
for i = 1, 10 do
  local v = H.visibilityFor(i / 10)
  T.check(v < prev, 'visibility falls as the strength rises (' .. i .. ')')
  prev = v
end

-- set / restore against a stubbed core_environment
local eng = { density = 4e-5, height = 1000 }
core_environment = {
  getState = function() return { fogDensity = eng.density * 1000, fogAtmosphereHeight = eng.height } end,
  setFogDensity = function(d) eng.density = d end,
  setFogAtmosphereHeight = function(h) eng.height = h end,
}
T.check(H.set(0.5), 'the engine took the fog')
T.near(eng.density, H.CONTRAST / H.visibilityFor(0.5), 1e-12, 'the density is written')
T.check(eng.height == H.HEIGHT_M and H.applied(), 'the height is written')
T.check(H.restore() and eng.density == 4e-5 and eng.height == 1000, 'restore puts the level\'s fog back')
T.check(not H.applied() and not H.restore(), 'restoring twice does nothing')

-- the slider back to 0 restores, through set()
H.set(0.8)
T.check(eng.height == H.HEIGHT_M, 'haze on')
H.set(0)
T.check(eng.density == 4e-5 and eng.height == 1000 and not H.applied(), 'slider to 0: the level\'s fog back')

-- update(): only in our level, re-written when something else reset the fog
os.clock = function() return 100 end
H.update(false, 0.5)
T.check(eng.density == 4e-5, 'other levels: untouched')
H.update(true, 0.5)
T.check(eng.height == H.HEIGHT_M, 'our level: the haze applied')
eng.density, eng.height = 4e-5, 1000               -- the weather module wrote LevelInfo again
os.clock = function() return 100 + H.CHECK_S / 2 end
H.update(true, 0.5)
T.check(eng.density == 4e-5, 'within CHECK_S: not re-checked')
os.clock = function() return 100 + H.CHECK_S * 2 end
H.update(true, 0.5)
T.near(eng.density, H.CONTRAST / H.visibilityFor(0.5), 1e-12, 'after CHECK_S: the haze re-applied')
H.update(true, 0)
T.check(eng.density == 4e-5 and eng.height == 1000, 'strength 0 in our level: the level\'s fog')
H.update(true, 0.5)
H.update(false, 0.5)
T.check(eng.density == 4e-5 and eng.height == 1000 and not H.applied(), 'leaving our level restores')

-- no engine at all (the level not loaded yet): no error, nothing claimed
core_environment = nil
T.check(H.set(0.5) == false and H.restore() == false, 'no core_environment: refused, not crashed')
T.done('test_haze')
