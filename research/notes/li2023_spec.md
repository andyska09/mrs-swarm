# Li 2023 — paper specification (audit reference)

Ground truth extracted from `research/papers/Li_2023_New_J._Phys._25_092001.md`
(Li, Li & Zhao, *Predator–prey survival pressure is sufficient to evolve swarming
behaviors*, New J. Phys. 25 092001, 2023).

**This file is the paper side only.** Nothing here is a statement about our
implementation. It exists so the audit has a fixed reference to compare against,
claim by claim. Every row cites its section / table / equation / figure.

---

## 1. Physical environment

### 1.1 Table 1 values (appendix B, table 1) — stated verbatim

| Parameter | Value | Unit |
|---|---|---|
| Mass of predator | 1 | kg |
| Mass of prey | 1 | kg |
| Max speed | 0.5 **or** 0.3 | m s⁻¹ |
| Max linear acc. | 1 | m s⁻² |
| Max angular vel. | 0.5 | rad s⁻¹ |
| Env edge length | 2 | m |
| Contact stiffness | 50 | N m⁻¹ |
| Drag coefficient | 2 | N·s m⁻¹ |
| Time step Δt | 0.1 | s |

The "0.5 or 0.3" is the speed-limit-ratio knob: 1:1 uses one value for both
species, 5:3 and 3:5 use 0.5 / 0.3 (§4.5). Drag and stiffness are also given in
body text (§2.2): "drag coefficient is set as 2 N·s m⁻¹ and the contact stiffness
coefficient is set as 50 N m⁻¹".

### 1.2 Dynamics (§2.2 eq 1, integrated per appendix B)

Continuous form (eq 1a–1c):

```
ẋ = v
v̇ = (a_F·h + f_d + f_a + f_b) / m_i
θ̇ = a_R
```

Discrete update, **in this exact order** (appendix B):

```
θ(t+1) = θ(t) + a_R·Δt
v(t+1) = v(t) + (a_F·h + f_d + f_a + f_b)·Δt / m_i
x(t+1) = x(t) + v(t)·Δt
```

Note the order: θ updates first; the velocity update uses `h`; the position
update uses `v(t)` — the *pre-update* velocity, as written.

| Force | Definition | Source |
|---|---|---|
| Active forward | `a_F·h`, `a_F ∈ R`, aligned with heading | §2.2 |
| Active rotation | `a_R`, direct heading rate, "within a threshold value" | §2.2, eq 1c |
| Drag `f_d` | opposite `v`, magnitude ∝ ‖v‖ (coeff 2) | §2.2 |
| Agent–agent elastic `f_a` | Hooke's law on overlap, summed over contacts `Σ_j f_a,j` | §2.2, appendix B |
| Agent–boundary elastic `f_b` | Hooke's law, summed, walls only | §2.2, appendix B |

"The velocity may not be aligned with the heading direction when a collision
happens" (§2.2) — velocity is a free 2-vector, not a speed scalar along `h`.

### 1.3 Action scaling (appendix B)

| Action | Range | Note |
|---|---|---|
| `a_F` | `[0, max linear acc.]` = `[0, 1]` | **forward only**, non-negative |
| `a_R` | `[−max angular vel., +max angular vel.]` = `[−0.5, 0.5]` | symmetric |

"`a_F` ranges from zero to its maximum linear acceleration, while `a_R` ranges
from negative maximum angular velocity to positive maximum angular velocity."
`a_F` is an **acceleration**; `a_R` is a **rate** applied directly to θ.

### 1.4 Boundaries (§2.1)

Two conditions, and the paper's central behavioral contrast rests on which is used:

| Kind | Behaviour | Used in |
|---|---|---|
| Periodic (torus) | wrap around, same velocity; approximates infinite space | §4.2 flocking, §4.5–4.7 ablations |
| Finite square with walls | cannot cross; walls have contact stiffness → `f_b` | §4.4 swirling |

### 1.5 Populations and arena

| Phase | n₀ predators | n₁ prey | Source |
|---|---|---|---|
| Training | 3 | 10 | §4.1 |
| Evaluation | 3 | **50** | §4.1 |

Spawn: "randomly spawn n₀ predators and n₁ prey" (algorithm 1) at "random
positions with random headings" (appendix B).

### 1.6 Observation (§2.3, appendix B)

Structure, in the paper's stated order:

```
[ agent's own pos., vel. and heading,
  relative pos. and headings of observed PREDATORS,
  relative pos. and headings of observed PREY ]
```

| Property | Value | Source |
|---|---|---|
| Metric range | disk of radius `R`; default `R = D` (= env size) | §2.3, §4.6 |
| Topological limit | **6 allies and 6 adversaries** (max) | §2.3 |
| Ordering | "reordered from the nearest to the farthest based on range" | appendix B |
| Overflow | farthest ones removed | §2.3, appendix B |
| Underflow | "masked out with zeros" | §2.3, appendix B |
| Own state | position, velocity, heading | §2.3, appendix B |
| Others | relative position, heading | §2.3, appendix B |

The topological limit of 6 is motivated by [27] — "each bird interacts on
average with six neighbors" (§2.3).

Dimension arithmetic implied: own (2 pos + 2 vel + 2 heading = 6) + 6 predators
× (2 rel pos + 2 heading) + 6 prey × 4 = 6 + 24 + 24 = **54**. The paper never
prints "54"; it only says `d_o` = "length of the observation vector"
(appendix B). The heading being 2-dim (unit vector) rather than 1-dim (angle) is
what makes the arithmetic land on 54 — the paper does not state which.

**Note the species ordering: predators before prey, for both species' observation
vectors.** The paper gives no ally-first rule.

---

## 2. Reward function (§3.4)

Per agent, per step:

```
prey:      r = −1  if in contact with a predator, else 0
predator:  r = +1  if in contact with a prey,     else 0
both:      r += −0.01·|a_F| − 0.1·|a_R|          (movement cost, "decorative")
walls:     r += −0.1  on contact with a boundary  (boundary-penalty case only)
```

Load-bearing details, all §3.4:

- Capture **is** contact: "the catch is represented by a contact between the two agents".
- **Prey are not removed after capture.** "prey agents are not removed from the
  simulation after being caught". Contact is "a continuous process where predators
  extract energy from the prey while engaged in 'eating' them" — so the ±1 is paid
  **every step the contact persists**, not once per capture event.
- "Upon separation, the prey's survival reward returns to zero."
- The movement cost is explicitly expected to cause laziness: "This reward
  function will cause the agent to exhibit laziness."
- The reward is **swarm-independent** by design — this is the paper's whole thesis.
  No cohesion, alignment, or separation term exists anywhere.
- Boundary penalty is `−0.1` and only applies "in the special case when boundaries
  exist"; §4.4 runs it both on and off.

---

## 3. Learning algorithm — MADDPG variant (§3.5, appendix B)

### 3.1 Structural choices (§3.5, three explicit deviations from standard MADDPG)

1. **Decentralized critic** `Q_i(o_i, a_i)` — local observation + own action only.
   No global state, no joint action (§3.2, §3.5).
2. **Parameter sharing within species, not across.** One actor + one critic per
   species; conspecifics share, adversaries do not (§3.1, §3.2, §3.5).
3. **One shared replay buffer per species.** `B₀` predators, `B₁` prey;
   conspecific experience is merged, not per-agent (§3.3, §3.5).

### 3.2 Networks (appendix B, figure 12)

| Property | Value |
|---|---|
| Architecture | deep feed-forward MLP |
| Hidden layers | 3 |
| Hidden size | 64 per layer |
| Activation | ReLU |
| Actor input / output | `d_o` / `d_a = 2` (`a_F`, `a_R`) |
| Critic input / output | `(o_i, a_i)` / scalar Q |

The paper does not state the actor's output squashing (tanh vs. sigmoid), where
the action enters the critic (input layer vs. later), or the optimizer.

### 3.3 Loss functions (§3.5 and appendix B — identical statements)

**Target value:**

```
y_i^j = r_i^j + γ · Q_i′(o_i′^j, a_i′^j)  ,   a_i′^j = μ_i′(o_i′^j)
```

**Critic loss** (minimized):

```
              1   S  (                     )²
L(θ_i^Q)  =  ───  Σ   y_i^j − Q_i(o_i^j, a_i^j)
              S  j=1
```

**Actor — sampled deterministic policy gradient** (ascended):

```
                1   S
∇_{θ_i^μ} J ≈  ───  Σ  ∇_{θ_i^μ} μ_i(o_i^j) · ∇_{μ_i(o_i^j)} Q_i(o_i^j, μ_i(o_i^j))
                S  j=1
```

**Soft update of both targets** (algorithm 1):

```
θ_i′^μ ← τ·θ_i^μ + (1 − τ)·θ_i′^μ
θ_i′^Q ← τ·θ_i^Q + (1 − τ)·θ_i′^Q
```

**There is no done/terminal mask in `y_i`.** The target is written
`y = r + γQ′` unconditionally — consistent with §3.4, where prey are never
removed and nothing terminates early. Episodes end only by hitting the length
limit.

### 3.4 Hyper-parameters (appendix B, table 2)

| Hyper-parameter | Value |
|---|---|
| Number of episodes | 2000 |
| Episode length | 100 |
| Number of hidden layers | 3 |
| Hidden layer size | 64 |
| Learning rate of actor | 1 × 10⁻⁴ |
| Learning rate of critic | 1 × 10⁻³ |
| Discount factor γ | 0.95 |
| Soft-update rate τ | 0.01 |
| Initial exploration rate ε | 0.1 |
| Initial noise rate N | 0.1 |
| Replay buffer size | 5 × 10⁵ |
| Batch size S | 256 |

An episode is 100 steps × Δt 0.1 s = **10 s of simulated time**; the full run is
2000 × 100 = **2 × 10⁵ environment steps per species**.

### 3.5 Exploration schedule (appendix B)

Two separate mechanisms, both decayed **per episode**:

```
ε  ← max(0.05, ε  − 5 × 10⁻⁵)      ε starts at 0.1   (probability of exploring)
N  ← max(0.05, N  − 5 × 10⁻⁵)      N starts at 0.1   (Gaussian action noise)
```

"The exploration rate ε is the probability that the agent will choose to explore
the environment instead of exploiting it."

Two things worth flagging as spec ambiguities:

- **ε appears only in appendix B, never in algorithm 1**, which shows only
  `a_i = μ_θi(o_i) + N_t`. How ε and `N_t` combine is unstated.
- At 5 × 10⁻⁵ per episode over 2000 episodes the total decay is 0.1, so both
  reach the floor of 0.05 at **episode 1000** and are constant thereafter.

---

## 4. Training loop (algorithm 1)

```
for species i in {0, 1}:
    init actor μ_i(θ_i^μ), critic Q_i(θ_i^Q)
    init targets  θ_i′^μ ← θ_i^μ ,  θ_i′^Q ← θ_i^Q

for episode = 1 .. M:                      # M = 2000
    randomly spawn n₀ predators, n₁ prey; get o_i
    for t = 1 .. max-episode-length:       # 100
        a_i = μ_θi(o_i) + N_t              # for all agents of species i
        execute a_i, receive r_i, o_i′
        store (o_i, a_i, r_i, o_i′) in B_i
        o_i ← o_i′
        sample mini-batch of S from B_i
        y_i = r_i + γ·Q_i′(o_i′, a_i′)
        update critic by minimizing L(θ_i^Q)
        update actor by sampled policy gradient
    soft-update targets                    # see note below
```

| Aspect | What the paper says | Source |
|---|---|---|
| Both species learn concurrently | yes — "learning and adapting their behaviors concurrently" | §3 |
| Update frequency | critic + actor updated **every environment step** | algorithm 1 |
| Batch source | one sample of S drawn per update, per species | algorithm 1 |
| Buffer warm-up | **not stated** | — |
| Gradient clipping | **not stated** | — |
| Optimizer | **not stated** (Adam is the MADDPG default of [25]) | — |
| Stopping criterion | "until a state of dynamic equilibrium is achieved between the predators and prey, such that neither party can obtain their future rewards by altering their respective policies" | appendix B |

**Soft-update placement is genuinely ambiguous in the transcript.** As typeset in
algorithm 1 the soft-update line sits at the same indent as the inner `for t`
loop's closing `end`, which reads as *once per episode*; standard MADDPG [25]
soft-updates every step. This is one to resolve by intent, not by the transcript.

---

## 5. Metrics (§4.1, eq 2–3)

**Degree of sparsity**, DoS ∈ [0, 1]:

```
            1    T   N
DoS  =  ───────  Σ   Σ   ‖x_j(t) − x_k(t)‖ ,    k = argmin_{k≠j} ‖x_j(t) − x_k(t)‖
          T·N·D t=1 j=1
```

**Degree of alignment**, DoA ∈ [0, 1]:

```
            1    T   N
DoA  =  ───────  Σ   Σ   ‖h_j(t) + h_k(t)‖ ,    same k as DoS
          2·T·N t=1 j=1
```

| Symbol | Definition | Source |
|---|---|---|
| `T` | episode length | eq 2 |
| `N` | "total number which is **equal to n₁ for prey**" | eq 2 |
| `D` | "environment size defined as the maximum possible distance for two agents" | eq 2 |
| `k` | nearest **conspecific** (`k ∈ {1..N}\j`) | eq 2 |

**`D` for the periodic case is given explicitly and is worth quoting:** "for a
periodic square environment with edge length 2, the largest possible distance is
√2." So `D = √2` on the torus with edge 2 — the wrap caps per-axis separation at
edge/2 = 1, giving √(1² + 1²) = √2. The paper does **not** state `D` for the
walled case; the maximum distance in a 2×2 box with no wrap is the full diagonal
2√2.

Both metrics are averaged over the whole episode (`1/T Σ_t`) — the paper draws
attention to this in §4.2 when contrasting figure 3 with figure 4.

**Reference value the paper itself computes:** for uniformly distributed heading
angles, `E[cos(φ/2)] = 2/π ≈ 0.64`, "quite close to the value read from figure 3"
(§4.2). The paper offers **no** corresponding random-configuration baseline for
DoS.

---

## 6. Claims the paper makes

Numbered for the audit. Each is a checkable assertion with a figure behind it.

### 6.1 Flocking, periodic boundary (§4.2, figures 2–5)

| # | Claim | Numbers | Source |
|---|---|---|---|
| C1 | Prey evolve cohesive flocking from survival pressure alone, over 2000 episodes | — | §4.2, fig 2(b) |
| C2 | **Episodic DoS falls 22% → ~19%** | 0.22 → 0.19 | §4.2, fig 3 |
| C3 | **Episodic DoA rises 0.65 → ~0.82** | 0.65 → 0.82 | §4.2, fig 3 |
| C4 | Initial DoA ≈ 2/π ≈ 0.64, matching random headings | 0.64–0.65 | §4.2 |
| C5 | Within an episode, under trained policies, DoS falls to ~15% and DoA rises to ~0.96 | 0.15 / 0.96 | §4.2, fig 4 |
| C6 | For established flocks DoS < 19% and DoA > 0.82 | — | §4.2 |
| C7 | Trained prey keep high DoA **with predators removed after training** — swarming, not mere herding; "comparable to the Vicsek model" | — | §4.2, fig 5(a) |
| C8 | Running average is 100 episodes; shaded band is 95% CI | — | §4.2, fig 3 |

### 6.2 Predator-side phenomena (§4.3, figures 6–7)

| # | Claim | Source |
|---|---|---|
| C9 | **Confusion effect** — a predator that meets a flock gives up the chase, slows, stagnates (visible as a shorter path at t = 30 → 33) | §4.3, fig 6 |
| C10 | **Dispersion tactic** — predators first move to the swarm centre to break it up, then pick off isolated prey | §4.3, fig 7(a) |
| C11 | **Marginal predation / edge effect** — captures concentrate on prey at the swarm periphery | §4.3, fig 7(b) |

### 6.3 Swirling, walled boundary (§4.4, figure 8)

| # | Claim | Source |
|---|---|---|
| C12 | In §4.4 **only**, predators are replaced by a scripted rule: rotate heading toward nearest prey, then move at maximum speed. Justification is that the boundary penalty slows learning. | §4.4 |
| C13 | Walls **without** boundary penalty → flocking essentially as in the periodic case, plus boundary aggregation ("fish in a fishbowl") | §4.4, fig 8(a) |
| C14 | Walls **with** the −0.1 boundary penalty → **swirling** (persistent circular milling) emerges | §4.4, fig 8(b) |
| C15 | Multiple flocks with opposed headings can cancel in a global order parameter — which is why DoS/DoA are defined locally (nearest neighbour) | §4.1, §4.2, figs 2(b), 8(b) |

### 6.4 Speed limit ratio (§4.5, figure 9)

Baseline for this section is the 1:1 flocking result (DoS 19%, DoA 0.82).

| # | Ratio `‖v₀‖max : ‖v₁‖max` | Claim | Source |
|---|---|---|---|
| C16 | 1 : 1 | baseline; DoS 19%, DoA 0.82 | §4.5 |
| C17 | 5 : 3 (predators faster) | **more** pronounced swarming: DoS 19% → 18%, DoA 0.82 → ~0.85 | §4.5, fig 9 |
| C18 | 3 : 5 (prey faster) | **higher convergence rate** of DoS and a **lower** final value ~17% — prey can evade while keeping formation | §4.5, fig 9 |

### 6.5 Perception range (§4.6, figure 10)

| # | Range | Claim | Source |
|---|---|---|---|
| C19 | `R = D` | default in all other simulations | §4.6 |
| C20 | `R = 2/3·D` | less pronounced flocking: larger DoS, smaller DoA (no numbers given) | §4.6, fig 10 |
| C21 | `R = 1/3·D` | notably worse: DoS 19% → ~20.5%, DoA 0.82 → 0.75 | §4.6, fig 10 |
| C22 | — | monotone trend: smaller perception ⇒ weaker flocking. "Perception range plays a crucial role." | §4.6 |

### 6.6 Number of predators (§4.7, figure 11)

| # | n₀ | Claim | Source |
|---|---|---|---|
| C23 | 3 | baseline | §4.7 |
| C24 | 1 | swarming still emerges, **slightly slower** — less survival pressure | §4.7, fig 11 |
| C25 | 0 | **DoS and DoA remain unchanged; no swarming emerges at all** | §4.7, figs 11, 5(b) |

C25 is the paper's control and its strongest single piece of evidence for the
title claim. It is also the discriminating test: n₀ = 0 flat and n₀ = 3 moving is
the whole result.

### 6.7 Headline

| # | Claim | Source |
|---|---|---|
| C26 | Survival pressure alone — a swarm-**independent** reward — suffices to evolve swarming; no handcrafted interaction rule or shaping term is needed | title, §5 |
| C27 | Mechanism hypothesis: flocking is "largely an outcome of **passive space extrusion and polarization** induced by predators" | §4.2, §5 |

---

## 7. What the paper does not specify

Every one of these is a free parameter we had to choose, and therefore a
candidate explanation for any divergence.

| Item | Status |
|---|---|
| **Agent radii** (predator and prey) | never stated — yet capture is contact, so radii set the entire capture rate |
| Initial velocity at spawn | never stated ("random positions with random headings" only) |
| Minimum spawn separation | never stated |
| Heading encoding in the obs (angle vs. unit vector) | never stated; unit vector is what makes `d_o` = 54 |
| Whether own position is absolute or arena-relative | "agent's own position" — absolute implied, unstated |
| Observation normalization | never mentioned |
| `D` for the walled case | only the periodic value (√2) is given |
| Actor output activation | never stated |
| Where the action enters the critic | never stated |
| Optimizer | never stated |
| Buffer warm-up before first update | never stated |
| Soft-update frequency (per step vs. per episode) | ambiguous as typeset in algorithm 1 |
| How ε and Gaussian noise `N_t` combine | ε appears only in appendix B, not in algorithm 1 |
| Number of seeds behind the 95% CI bands | never stated |
| Max speed enforcement (clip vs. drag equilibrium) | "max speed" is tabulated but no clipping step appears in the appendix B integration order |

The last one is worth its own line: appendix B's update has **no speed clamp**.
With drag coefficient 2, mass 1 and max `a_F` = 1, the drag equilibrium is
‖v‖ = a_F/2 = 0.5 m s⁻¹ — exactly the tabulated "max speed". So on the 1:1
setting the speed limit may be emergent from the drag, not enforced. That
reading does not extend to 0.3, which would need either a different drag
coefficient or an explicit clamp.
