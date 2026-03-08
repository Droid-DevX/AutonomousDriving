import math
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from collections import deque


#  pixel font
_FONT5 = {
    "N": ["10001","11001","10101","10011","10001"],
    "P": ["11100","10010","11100","10000","10000"],
    "C": ["01110","10000","10000","10000","01110"],
    "H": ["10001","10001","11111","10001","10001"],
    "I": ["11111","00100","00100","00100","11111"],
    "T": ["11111","00100","00100","00100","00100"],
    "O": ["01110","10001","10001","10001","01110"],
    "V": ["10001","10001","10001","01010","00100"],
    "E": ["11111","10000","11110","10000","11111"],
    "R": ["11110","10001","11110","10100","10010"],
    "A": ["01110","10001","11111","10001","10001"],
    "K": ["10001","10010","11100","10010","10001"],
    "D": ["11100","10010","10001","10010","11100"],
    " ": ["00000","00000","00000","00000","00000"],
    "1": ["00100","01100","00100","00100","01110"],
    "2": ["01110","10001","00110","01000","11111"],
    "3": ["11110","00001","01110","00001","11110"],
}

def _draw_pixel_text(frame, cx, top_y, text, color, scale=1):
    """Draw pixel text centred at cx, top edge at top_y, onto frame in-place."""
    chars  = [_FONT5.get(ch, _FONT5[" "]) for ch in text.upper()]
    cw     = 5 * scale
    gap    = scale
    total  = len(chars) * cw + (len(chars)-1) * gap
    lx     = cx - total // 2
    fh, fw = frame.shape[:2]
    for ci, bmp in enumerate(chars):
        ox = lx + ci * (cw + gap)
        for row in range(5):
            for col in range(5):
                if bmp[row][col] == "1":
                    for dy in range(scale):
                        for dx in range(scale):
                            fy = top_y - (5 - row) * scale + dy
                            fx = ox + col * scale + dx
                            if 0 <= fy < fh and 0 <= fx < fw:
                                frame[fy, fx] = color




class _NPC:
    """Holds all mutable state for one NPC car."""

    # Colours per NPC index so they're visually distinct
    BODY_COLOURS = [
        ([210, 30,  30 ], [30,  30,  30 ]),   
        ([30,  80,  210], [20,  20,  60 ]),   
        ([30,  180, 30 ], [20,  60,  20 ]),   
    ]

    def __init__(self, idx: int):
        self.idx            = idx
        self.tile_f         = 0.0   
        self.x              = 0.0
        self.y              = 0.0
        self.hit            = False
        self.ot_state        = "WAITING"   
        self.player_is_ahead = False      
        self.spawn_grace     = 0         
        self.active          = False      

        colours = self.BODY_COLOURS[idx % len(self.BODY_COLOURS)]
        self.roof_colour = colours[0]
        self.body_colour = colours[1]


# main wrapper

class ObstacleWrapper(Wrapper):

    NPC_TILES_PER_STEP = 0.05  
    NPC_COUNT          = 3      

 
    NPC_SPAWN_STEPS    = [0, 400, 700]

   
    SPAWN_TILES_AHEAD  = 20
    SPAWN_TILE_GAP     = 15   

  
    COLLISION_DIST  = 1.5   
    ENGAGE_DIST     = 5.0   
    SURPASS_DIST    = 3.0    
    MIN_AHEAD_TILES = 6      

    def __init__(self, env, npc_penalty=-5.0, overtake_reward=15.0, npc_count=None):
        super().__init__(env)
        self.npc_penalty      = npc_penalty
        self.overtake_reward  = overtake_reward
        self._npc_count       = npc_count if npc_count is not None else self.NPC_COUNT

        self._npcs: list[_NPC] = []
        self._total_overtakes  = 0
        self._step_count       = 0

    # helpers 

    def _raw(self):
        return self.env.unwrapped

    def _track(self):
        try:    return self._raw().track
        except: return None

    def _car_pos(self):
        try:
            u = self._raw()
            return float(u.car.hull.position.x), float(u.car.hull.position.y)
        except:
            return 0.0, 0.0

    def _zoom(self):
        try:
            z = self._raw().zoom
            return float(z) if z and float(z) > 0.1 else 2.7
        except:
            return 2.7

    def _tile_pos(self, idx):
        t = self._track()
        if t is None: return 0.0, 0.0
        r = t[int(idx) % len(t)]
        return float(r[2]), float(r[3])

    def _nearest_tile_idx(self, wx, wy):
        t = self._track()
        if t is None: return 0
        best_i, best_d = 0, float("inf")
        for i, r in enumerate(t):
            d = (r[2] - wx) ** 2 + (r[3] - wy) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    #  NPC init + movement

    def _spawn_npc(self, npc):
        """Place an NPC on the track ahead of the player and mark it active."""
        t = self._track()
        if t is None: return
        n = len(t)
        cx, cy       = self._car_pos()
        car_tile     = self._nearest_tile_idx(cx, cy)
        spawn_tile   = (car_tile + self.SPAWN_TILES_AHEAD) % n
        npc.tile_f   = float(spawn_tile)
        npc.x, npc.y = self._tile_pos(spawn_tile)
        npc.active          = True
        npc.hit             = False
        npc.ot_state        = "WAITING"
        npc.player_is_ahead = False
        npc.spawn_grace     = 100   
        print(f"  [NPC {npc.idx}] spawned at step {self._step_count}, "
              f"tile {spawn_tile}/{n} ({npc.x:.1f}, {npc.y:.1f})")

    def _init_npcs(self):
        """Create all NPC slots; only NPC 0 is activated immediately."""
        spawn_steps = self.NPC_SPAWN_STEPS
        self._npcs = [_NPC(i) for i in range(self._npc_count)]
        # Spawn only NPC 0 at the start
        self._spawn_npc(self._npcs[0])
        upcoming = " | ".join(
            f"NPC {i+1} @ step {spawn_steps[i]}"
            for i in range(1, self._npc_count)
            if i < len(spawn_steps)
        )
        print(f"  Delayed spawns: {upcoming}")

    def _advance_npcs(self):
        t = self._track()
        if t is None: return
        n = len(t)
        for npc in self._npcs:
            if not npc.active: continue          
            npc.tile_f   = (npc.tile_f + self.NPC_TILES_PER_STEP) % n
            npc.x, npc.y = self._tile_pos(npc.tile_f)

    # overtake + collision (per NPC) 

    def _check_events(self) -> float:
        t = self._track()
        if t is None: return 0.0
        n     = len(t)
        delta = 0.0
        self._step_penalty = 0.0   
        self._step_reward  = 0.0   

        cx, cy      = self._car_pos()
        player_tile = self._nearest_tile_idx(cx, cy)

        for npc in self._npcs:
            if not npc.active: continue    
            dist     = math.hypot(npc.x - cx, npc.y - cy)
            npc_tile = int(npc.tile_f) % n

            #  Collision 
            if not npc.hit and dist < self.COLLISION_DIST:
                npc.hit = True
                delta              += self.npc_penalty
                self._step_penalty += self.npc_penalty
                print(f"   [NPC {npc.idx}] Collision! dist={dist:.2f}  {self.npc_penalty:+.0f}")
            elif npc.hit and dist > self.COLLISION_DIST * 4:
                npc.hit = False

          
            tiles_ahead = (player_tile - npc_tile) % n

            player_is_behind = tiles_ahead == 0 or tiles_ahead > n // 2

            
            player_is_ahead = (
                self.MIN_AHEAD_TILES <= tiles_ahead <= n // 2
                and dist > self.SURPASS_DIST
            )

           
            if npc.spawn_grace > 0:
                npc.spawn_grace -= 1
                npc.player_is_ahead = False
            else:
                npc.player_is_ahead = player_is_ahead

            if npc.ot_state == "WAITING":
               
                if dist < self.ENGAGE_DIST and player_is_behind:
                    npc.ot_state = "ARMED"
                    npc.hit      = False   
                    print(f"  [NPC {npc.idx}] ARMED — dist={dist:.1f} tiles_ahead={tiles_ahead}")

            elif npc.ot_state == "ARMED":
                if self._step_count % 10 == 0:
                    print(f"  [NPC {npc.idx}] ARMED — tiles_ahead={tiles_ahead} dist={dist:.2f} "
                          f"(need >={self.MIN_AHEAD_TILES} tiles AND dist>{self.SURPASS_DIST})")

                if player_is_ahead:
                   
                    npc.ot_state = "AWARDED"
                    if not npc.hit:
                        self._total_overtakes += 1
                        delta                 += self.overtake_reward
                        self._step_reward     += self.overtake_reward
                        print(f"  [NPC {npc.idx}] Overtake #{self._total_overtakes}! "
                              f"+{self.overtake_reward:.0f}  tiles_ahead={tiles_ahead}  dist={dist:.1f}")
                    else:
                        print(f"    [NPC {npc.idx}] Passed after collision — no bonus")

               
                elif player_is_behind and dist > self.ENGAGE_DIST * 2:
                    npc.ot_state = "WAITING"
                    print(f"  [NPC {npc.idx}] disarmed — player retreated (dist={dist:.1f})")

            elif npc.ot_state == "AWARDED":
                
                if dist < self.ENGAGE_DIST and player_is_behind:
                    npc.ot_state = "WAITING"
                    npc.hit      = False
                    print(f"  [NPC {npc.idx}] RESET → WAITING (NPC caught up, dist={dist:.1f})")

        return delta

   

    def _draw_on_obs(self, obs):
        try:
            cx, cy = self._car_pos()
            zoom   = self._zoom()
            frame  = obs.copy()

            for npc in self._npcs:
                if not npc.active: continue
                px = (npc.x - cx) * zoom + 48.0
                py = (cy - npc.y) * zoom + 48.0
                if not (-5 <= px < 101 and -5 <= py < 101):
                    continue
                hw, hh = 1, 2
                ipx, ipy = int(round(px)), int(round(py))
                y0 = max(0, ipy - hh);  y1 = min(96, ipy + hh + 1)
                x0 = max(0, ipx - hw);  x1 = min(96, ipx + hw + 1)
                if y1 <= y0 or x1 <= x0: continue
                frame[y0:y1, x0:x1] = [80, 80, 80] if npc.hit else npc.body_colour
                if not npc.hit:
                    frame[y0:y0 + max(1, (y1 - y0) // 3), x0:x1] = npc.roof_colour

                if npc.player_is_ahead:
                    label_text  = "OVERTAKEN"
                    label_color = [0, 255, 80]
                elif npc.hit:
                    label_text  = "NPC HIT"
                    label_color = [255, 60, 60]
                else:
                    label_text  = f"NPC {npc.idx + 1}"
                    label_color = [255, 255, 0]
                _draw_pixel_text(frame, int(round(px)), y0 - 2, label_text, label_color, scale=1)

            return frame
        except:
            return obs

   

    def draw_on_render(self, frame):
        try:
            cx, cy  = self._car_pos()
            zoom    = self._zoom()
            fh, fw  = frame.shape[:2]
            out     = frame.copy()
            sc      = fw / 96.0

            for npc in self._npcs:
                if not npc.active: continue
                px = (npc.x - cx) * zoom + fw / 2.0
                py = (cy - npc.y) * zoom + fh / 2.0
                if not (0 <= px < fw and 0 <= py < fh): continue
                hw = max(2, int(2 * sc));  hh = max(3, int(3.5 * sc))
                ipx, ipy = int(round(px)), int(round(py))
                y0 = max(0, ipy - hh); y1 = min(fh, ipy + hh)
                x0 = max(0, ipx - hw); x1 = min(fw, ipx + hw)
                if y1 <= y0 or x1 <= x0: continue
                out[y0:y1, x0:x1] = [80, 80, 80] if npc.hit else npc.body_colour
                if not npc.hit:
                    out[y0:y0 + max(1, (y1 - y0) // 3), x0:x1] = npc.roof_colour
                    ww = max(1, int(sc)); wh = max(1, int(sc))
                    out[y0:y0+wh, x0:x0+ww]   = [5, 5, 5]
                    out[y0:y0+wh, x1-ww:x1]   = [5, 5, 5]
                    out[y1-wh:y1, x0:x0+ww]   = [5, 5, 5]
                    out[y1-wh:y1, x1-ww:x1]   = [5, 5, 5]

                if npc.player_is_ahead:
                    label_text  = "OVERTAKEN"
                    label_color = [0, 255, 80]
                elif npc.hit:
                    label_text  = "NPC HIT"
                    label_color = [255, 60, 60]
                else:
                    label_text  = f"NPC {npc.idx + 1}"
                    label_color = [255, 255, 0]
                ps = max(2, int(sc))
                _draw_pixel_text(out, int(round(px)), y0 - ps - 2, label_text, label_color, scale=ps)

            return out
        except:
            return frame

    # Gymnasium API

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count      = 0
        self._total_overtakes = 0
        self._init_npcs()
        return self._draw_on_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Advance all active NPCs
        self._advance_npcs()

        #  Delayed NPC spawns
        for i, npc in enumerate(self._npcs):
            if not npc.active:
                spawn_at = self.NPC_SPAWN_STEPS[i] if i < len(self.NPC_SPAWN_STEPS) else 99999
                if self._step_count >= spawn_at:
                    self._spawn_npc(npc)

        extra   = self._check_events()
        reward += extra

        # Store penalty and reward independently so both can be non-zero in same step
        info["npc_penalty"]     = self._step_penalty   
        info["npc_reward"]      = self._step_reward    
        info["npc_hit"]         = any(npc.hit for npc in self._npcs)
        info["total_overtakes"] = self._total_overtakes
        self._step_count += 1

        if self._step_count % 200 == 0:
            cx, cy = self._car_pos()
            t      = self._track()
            if t:
                n  = len(t)
                pt = self._nearest_tile_idx(cx, cy)
                for npc in self._npcs:
                    nt   = int(npc.tile_f) % n
                    dist = math.hypot(npc.x - cx, npc.y - cy)
                    print(f"  [step {self._step_count}][NPC {npc.idx}] "
                          f"dist={dist:.1f} player_tile={pt} npc_tile={nt} "
                          f"tiles_ahead={(pt - nt) % n} state={npc.ot_state}")

        return self._draw_on_obs(obs), reward, terminated, truncated, info




def run_visual_test(model, make_env_fn, n_episodes=5, window_size=700, fps=30):
    import pygame

    env = make_env_fn()()
    ow, e = None, env
    while e is not None:
        if isinstance(e, ObstacleWrapper): ow = e; break
        e = getattr(e, "env", None)
    if ow is None:
        raise RuntimeError("ObstacleWrapper not found in env chain")

    pygame.init()
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("CarRacing — Overtake the NPCs!")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 16)

    def make_stack(buf):
        return np.transpose(np.concatenate(list(buf), axis=2), (2, 0, 1))[None]

    print(f"Running {n_episodes} episodes\n")
    print(f"NPCs: {ow._npc_count} cars | "
          f"{ObstacleWrapper.NPC_TILES_PER_STEP} tiles/step | "
          f"spawn: first at {ObstacleWrapper.SPAWN_TILES_AHEAD} tiles ahead, "
          f"gap={ObstacleWrapper.SPAWN_TILE_GAP}\n")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        buf    = deque([obs] * 4, maxlen=4)
        ep_rew = pen = ot_rew = steps = 0
        done   = False

        while not done:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    env.close(); pygame.quit(); return

            action, _ = model.predict(make_stack(buf), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(
                np.array(action[0], dtype=np.float64))
            done    = terminated or truncated
            buf.append(obs)
            ep_rew += float(reward)
            pen    += float(info.get("npc_penalty", 0.0))   
            ot_rew += float(info.get("npc_reward",  0.0))   
            steps  += 1

            raw     = ow._raw().render()
            display = ow.draw_on_render(raw) if raw is not None and raw.size > 0 else obs
            surf    = pygame.surfarray.make_surface(np.transpose(display, (1, 0, 2)))
            screen.blit(pygame.transform.scale(surf, (window_size, window_size)), (0, 0))

            ot = info.get("total_overtakes", 0)
            hit_now = info.get("npc_hit", False)
            screen.blit(font.render(
                f"Ep {ep+1}/{n_episodes}   Reward: {ep_rew:7.1f}",
                True, (255, 255, 255)), (8, 8))
            screen.blit(font.render(
                f"Overtakes:{ot}  OT+:{ot_rew:+.0f}  Pen:{pen:.0f}  "
                f"Hit:{'YES' if hit_now else 'no'}",
                True, (100, 255, 100) if ot > 0 else (255, 200, 80)), (8, 28))
            # Flash reward/penalty event for 1 step
            if info.get("npc_reward", 0) > 0:
                screen.blit(font.render(
                    f"+{info['npc_reward']:.0f} OVERTAKE!", True, (0, 255, 80)), (8, 68))
            if info.get("npc_penalty", 0) < 0:
                screen.blit(font.render(
                    f"{info['npc_penalty']:.0f} COLLISION!", True, (255, 60, 60)), (8, 68))

            # Show per-NPC state (inactive NPCs show countdown to spawn)
            def _npc_label(i, npc):
                if not npc.active:
                    spawn_at = ObstacleWrapper.NPC_SPAWN_STEPS[i] if i < len(ObstacleWrapper.NPC_SPAWN_STEPS) else 0
                    remaining = max(0, spawn_at - steps)
                    return f"NPC{i+1}:in {remaining}"
                return f"NPC{i+1}:{npc.ot_state[:3]}"
            states = " | ".join(_npc_label(i, npc) for i, npc in enumerate(ow._npcs))
            screen.blit(font.render(states, True, (180, 180, 255)), (8, 48))
            pygame.display.flip()
            clock.tick(fps)

        print(f"Ep {ep+1:2d} | Reward:{ep_rew:8.2f} | "
              f"Overtakes:{info.get('total_overtakes', 0)} | "
              f"OT_Reward:{ot_rew:+.0f} | Penalty:{pen:.1f} | Steps:{steps}")

    env.close()
    pygame.quit()
    print("\nDone.")
