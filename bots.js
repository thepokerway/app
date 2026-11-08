(function() {
  'use strict';

  const R = Engine.RANK_VALUE;
  const CLASS_PRO = 'PRO';

  // ========================================================================
  // BLOCK 0 — SHARED CORE: TYPES, PROFILES, CONSTANTS
  // ========================================================================

  const TYPES = {
    PRO_GTO:    { key: 'PRO_GTO',    cls: CLASS_PRO },
    PRO_SOLVER: { key: 'PRO_SOLVER', cls: CLASS_PRO },
    PRO_ICM:    { key: 'PRO_ICM',    cls: CLASS_PRO },
    PRO_LAG:    { key: 'PRO_LAG',    cls: CLASS_PRO }
  };

  const PROFILES = {
    PRO_GTO:    { aggression: 0.68, looseness: 0.42, bluff: 0.22, callThreshold: 0.55, icmSensitivity: 0.70, positionAdaptivity: 1.00, limpHabit: 0.10 },
    PRO_SOLVER: { aggression: 0.78, looseness: 0.55, bluff: 0.34, callThreshold: 0.45, icmSensitivity: 0.55, positionAdaptivity: 1.15, limpHabit: 0.06 },
    PRO_ICM:    { aggression: 0.72, looseness: 0.45, bluff: 0.28, callThreshold: 0.52, icmSensitivity: 1.30, positionAdaptivity: 1.05, limpHabit: 0.08 },
    PRO_LAG:    { aggression: 0.92, looseness: 0.72, bluff: 0.48, callThreshold: 0.32, icmSensitivity: 0.35, positionAdaptivity: 1.25, limpHabit: 0.04 }
  };

  const STACK_DEPTHS = [
    { key: '100+', min: 100, max: Infinity },
    { key: '80-60', min: 60, max: 100 },
    { key: '60-40', min: 40, max: 60 },
    { key: '40-30', min: 30, max: 40 },
    { key: '30-25', min: 25, max: 30 },
    { key: '25-20', min: 20, max: 25 },
    { key: '20-15', min: 15, max: 20 },
    { key: '15-10', min: 10, max: 15 },
    { key: '<10', min: 0, max: 10 }
  ];

  const POSITION_MULTIPLIERS = {
    UTG: 1.00, UTG1: 1.10, LJ: 1.25, HJ: 1.45,
    CO: 1.80, BTN: 3.00, SB: 2.50, BB: 2.20
  };

  const BASE_OPEN_THRESHOLDS = {
    UTG: 0.72, UTG1: 0.68, LJ: 0.62, HJ: 0.55,
    CO: 0.45, BTN: 0.32, SB: 0.38, BB: 0.60
  };

  const NASH_PUSH_TABLE = {
    UTG:  { 5: 0.22, 8: 0.18, 10: 0.16, 12: 0.14, 15: 0.12, 20: 0.10, 25: 0.09 },
    UTG1: { 5: 0.25, 8: 0.21, 10: 0.18, 12: 0.16, 15: 0.14, 20: 0.12, 25: 0.10 },
    LJ:   { 5: 0.30, 8: 0.26, 10: 0.22, 12: 0.19, 15: 0.16, 20: 0.13, 25: 0.11 },
    HJ:   { 5: 0.36, 8: 0.31, 10: 0.26, 12: 0.22, 15: 0.18, 20: 0.15, 25: 0.12 },
    CO:   { 5: 0.44, 8: 0.38, 10: 0.32, 12: 0.27, 15: 0.22, 20: 0.18, 25: 0.14 },
    BTN:  { 5: 0.55, 8: 0.48, 10: 0.42, 12: 0.35, 15: 0.30, 20: 0.24, 25: 0.19 },
    SB:   { 5: 0.68, 8: 0.60, 10: 0.52, 12: 0.44, 15: 0.38, 20: 0.30, 25: 0.24 },
    BB:   { 5: 0.50, 8: 0.42, 10: 0.36, 12: 0.30, 15: 0.25, 20: 0.20, 25: 0.16 }
  };

  const PRIORS = {
    rfi: 0.25, threeBet: 0.08, foldTo3Bet: 0.50, coldCall: 0.15,
    fourBet: 0.05, foldTo4Bet: 0.55, call4Bet: 0.40, fiveBet: 0.10, cold4Bet: 0.02,
    squeeze: 0.08, foldToSqueeze: 0.55, limp: 0.08, foldToLimp: 0.40,
    resteal: 0.10, foldToResteal: 0.55, foldStealToResteal: 0.50,
    isoRaise: 0.12, callIsoRaise: 0.35, raiseIsoRaise: 0.10,
    steal: 0.35, foldBbToStealBtn: 0.45, foldBbToStealCo: 0.50, foldBbToStealSb: 0.55,
    foldSbToStealBtn: 0.60, foldSbToStealCo: 0.65,
    bbDefendVsUtg: 0.20, bbDefendVsUtg1: 0.22, bbDefendVsLj: 0.28, bbDefendVsHj: 0.34,
    bbDefendVsCo: 0.42, bbDefendVsBtn: 0.52, bbDefendVsSb: 0.55,
    sbRfiVsBb: 0.38, bbDefendVsSbRfi: 0.55, sbFoldVsBb3Bet: 0.72,
    positionalAwareness: 0.50
  };

  const SHRINK_K = {
    default: 12, rfi: 10, threeBet: 12, foldTo3Bet: 14, fourBet: 10,
    foldTo4Bet: 12, bbDefend: 15, sbRfi: 12, steal: 10, resteal: 12, isoRaise: 10
  };

  // Confidence thresholds for stat usage
  const STAT_CONFIDENCE = {
    IGNORE: 0.25,
    MICRO: 0.45,
    MODERATE: 0.70
  };

  // ========================================================================
  // BLOCK 1 — HAND EVALUATION & PREFLOP STRENGTH
  // ========================================================================

  function preflopStrength(card1, card2) {
    const v1 = R[card1.rank], v2 = R[card2.rank];
    const hi = Math.max(v1, v2), lo = Math.min(v1, v2);
    const suited = card1.suit === card2.suit;
    if (v1 === v2) return 0.50 + (hi - 2) / 12 * 0.50;
    let s = (hi - 2) / 12 * 0.55 + (lo - 2) / 12 * 0.25;
    if (suited) s += 0.08;
    if (hi === 14) s += 0.05;
    if (Math.abs(hi - lo) === 1) s += 0.04;
    if (Math.abs(hi - lo) === 2) s += 0.02;
    return Math.min(1, Math.max(0, s));
  }

  function getStackDepth(stackBB) {
    for (const d of STACK_DEPTHS) {
      if (stackBB >= d.min && stackBB < d.max) return d.key;
    }
    return '<10';
  }

  function getPositionMultiplier(position) {
    return POSITION_MULTIPLIERS[position] || 1.0;
  }

  function getBlockerScore(card1, card2) {
    const r1 = card1.rank, r2 = card2.rank;
    const v1 = R[r1], v2 = R[r2];
    let score = 0;
    if (v1 === 14 && v2 >= 2 && v2 <= 5 && card1.suit === card2.suit) score += 0.15;
    if (v1 === 13 && v2 >= 10 && v2 <= 12 && card1.suit === card2.suit) score += 0.05;
    if ((v1 === 12 && v2 === 11) || (v1 === 11 && v2 === 10)) score -= 0.10;
    if (v1 === v2 && v1 <= 8) score -= 0.20;
    if (v1 === 14 && v2 === 9 && card1.suit !== card2.suit) score += 0.10;
    return score;
  }

  function equityRealization(card1, card2, position, multiway) {
    let mult = 1.0;
    const v1 = R[card1.rank], v2 = R[card2.rank];
    const suited = card1.suit === card2.suit;
    const isConnector = Math.abs(v1 - v2) === 1;
    const isSmallPair = v1 === v2 && v1 <= 8;
    const isBroadwayOffsuit = !suited && v1 >= 10 && v2 >= 10;

    if (position === 'BTN' || position === 'CO') mult *= 1.08;
    else if (position === 'SB' || position === 'BB') mult *= 0.88;
    else mult *= 0.97;

    if (suited) mult += 0.05;
    if (isConnector) mult += 0.03;
    if (isBroadwayOffsuit) mult -= 0.08;
    if (isSmallPair && (position === 'SB' || position === 'BB')) mult -= 0.05;
    if (multiway > 2) mult -= 0.06;

    return Math.max(0.70, Math.min(1.15, mult));
  }

  // ========================================================================
  // BLOCK 2 — OPPONENT MODEL (SESSION STATISTICS + BAYESIAN SHRINKAGE)
  // ========================================================================

  const opponentStats = new Map();

  const OPP_KEYS = [
    'rfi', 'threeBet', 'foldTo3Bet', 'coldCall', 'squeeze', 'foldToSqueeze',
    'limp', 'resteal', 'foldToResteal', 'foldStealToResteal',
    'push', 'callPush',
    'fourBet', 'foldTo4Bet', 'call4Bet', 'fiveBet', 'cold4Bet',
    'steal', 'foldBbToStealBtn', 'foldBbToStealCo', 'foldBbToStealSb',
    'foldSbToStealBtn', 'foldSbToStealCo',
    'isoRaise', 'callIsoRaise', 'raiseIsoRaise',
    'squeezeVs1', 'squeezeVs2', 'foldToSqueezeVs1', 'foldToSqueezeVs2',
    'bbDefendVsUtg', 'bbDefendVsUtg1', 'bbDefendVsLj', 'bbDefendVsHj',
    'bbDefendVsCo', 'bbDefendVsBtn', 'bbDefendVsSb',
    'bb3BetVsUtg', 'bb3BetVsUtg1', 'bb3BetVsLj', 'bb3BetVsHj',
    'bb3BetVsCo', 'bb3BetVsBtn', 'bb3BetVsSb',
    'sbDefendVsCo', 'sbDefendVsBtn', 'sb3BetVsCo', 'sb3BetVsBtn',
    'sbRfiVsBb', 'sbLimpVsBb', 'bbDefendVsSbRfi', 'bb3BetVsSbRfi',
    'sbFoldVsBb3Bet', 'sbCallVsBb3Bet', 'sb4BetVsBb3Bet'
  ];

  function getOrCreateStats(playerId) {
    if (!opponentStats.has(playerId)) {
      const s = {
        handsPlayed: 0, showdowns: 0, showdownWins: 0,
        limpTrapCount: 0, adaptationDetected: false, positionalAwareness: 0.5
      };
      for (const k of OPP_KEYS) {
        s[k] = 0;
        s[k + 'Opportunities'] = 0;
      }
      opponentStats.set(playerId, s);
    }
    return opponentStats.get(playerId);
  }

  function shrink(observed, opportunities, priorKey, kOverride) {
    const prior = PRIORS[priorKey] !== undefined ? PRIORS[priorKey] : 0.30;
    const k = kOverride || SHRINK_K[priorKey] || SHRINK_K.default;
    const opp = Math.max(0, opportunities || 0);
    const obs = Math.max(0, Math.min(observed || 0, opp));
    const adjusted = (obs + prior * k) / (opp + k);
    const confidence = opp / (opp + k);
    return { value: adjusted, raw: opp > 0 ? obs / opp : prior, confidence: confidence, sampleSize: opp };
  }

  function getShrunk(playerId, key) {
    const stats = opponentStats.get(playerId);
    if (!stats) {
      const prior = PRIORS[key] !== undefined ? PRIORS[key] : 0.30;
      return { value: prior, raw: prior, confidence: 0, sampleSize: 0 };
    }
    return shrink(stats[key] || 0, stats[key + 'Opportunities'] || 0, key);
  }

  function classifyOpponent(stats) {
    if (stats.handsPlayed < 10) return 'unknown';
    const vpip = shrink(stats.coldCall + stats.limp, stats.coldCallOpportunities || stats.limp + stats.coldCall + 1, 'coldCall').value;
    const pfr = shrink(stats.rfi, stats.rfiOpportunities || 1, 'rfi').value;
    const tb = shrink(stats.threeBet, stats.threeBetOpportunities || 1, 'threeBet').value;
    const f3b = shrink(stats.foldTo3Bet, stats.foldTo3BetOpportunities || 1, 'foldTo3Bet').value;

    if (vpip < 0.20 && pfr < 0.15 && tb < 0.04) return 'nit';
    if (vpip > 0.45 && pfr < 0.14) return 'station';
    if (vpip > 0.40 && pfr > 0.28 && tb > 0.12) return 'maniac';
    if (vpip > 0.28 && pfr > 0.22 && tb > 0.08) return 'lag';
    if (f3b > 0.68) return 'nit';
    if (vpip > 0.20 && pfr > 0.16) return 'tag';
    return 'tag';
  }

  function getOpponentType(playerId) {
    const stats = opponentStats.get(playerId);
    if (!stats) return 'unknown';
    return classifyOpponent(stats);
  }

  function updateOpponentStats(playerId, action, context) {
    const stats = getOrCreateStats(playerId);
    if (action === 'hand_played') { stats.handsPlayed++; return; }
    if (action === 'showdown') { stats.showdowns++; return; }
    const oppKey = action + 'Opportunities';
    if (stats[oppKey] !== undefined) stats[oppKey]++;
    if (stats[action] !== undefined) stats[action]++;
    if (action === 'rfi' && context && context.position === 'BTN') stats._btnOpen = (stats._btnOpen || 0) + 1;
    if (action === 'rfi' && context && context.position === 'UTG') stats._utgOpen = (stats._utgOpen || 0) + 1;
    if (stats._btnOpen && stats._utgOpen) {
      stats.positionalAwareness = Math.min(1, stats._btnOpen / Math.max(1, stats._utgOpen) / 2);
    }
  }

  // ========================================================================
  // BLOCK 3 — ICM / BUBBLE FACTOR
  // ========================================================================

  function calculateBubbleFactor(myStackBB, opponentStackBB, playersLeft, itmCount) {
    if (playersLeft <= itmCount) return 1.0;
    const onBubble = (playersLeft - itmCount) <= 2;
    if (!onBubble) return 1.0 + Math.max(0, (itmCount / Math.max(1, playersLeft)) - 0.3) * 0.5;
    if (myStackBB < 15) return 1.4;
    if (myStackBB > 40) return 1.6;
    return 1.8;
  }

  function getICMPressure(playersLeft, itmCount) {
    if (playersLeft <= 3) return 'final_table';
    if (playersLeft <= itmCount + 2) return 'bubble';
    return 'none';
  }

  function getPayoutStructure(playersLeft, itmCount) {
    if (itmCount <= 3) return 'steep';
    if (itmCount <= 8) return 'standard';
    return 'flat';
  }

  function adjustThresholdForICM(baseThreshold, bubbleFactor, icmSensitivity) {
    const adjustment = (bubbleFactor - 1.0) * icmSensitivity * 0.15;
    return Math.min(0.98, baseThreshold + adjustment);
  }

  // ========================================================================
  // BLOCK 3-B — CONTEXT DETECTION & STAT APPLICATION HELPERS
  // ========================================================================

  // Determine the exact preflop situation from engine-provided data.
  // Engine passes: toCall, currentBet, bb, contextType (already computed), numLimpers,
  // isRealRaise, raiserPosition, facingAllin, facing3Bet, facing4Bet, position.
  function detectContext(ctx) {
    const street = ctx.street;
    const position = ctx.position;
    const toCall = ctx.toCall || 0;
    const bb = ctx.bb || 1;
    const currentBet = ctx.currentBet || 0;

    if (street !== 'preflop') {
      if (toCall > 0) {
        if (ctx.facingAllin) return 'facing_allin_postflop';
        return 'facing_bet_postflop';
      }
      return 'unopened_postflop';
    }

    // Engine-provided contextType is authoritative if present
    if (ctx.contextType) {
      switch (ctx.contextType) {
        case 'unopened': return 'unopened';
        case 'facing_limp': return 'facing_limp';
        case 'bb_check_option': return 'bb_check_option';
        case 'facing_raise': return 'facing_raise';
        case 'facing_3bet': return 'facing_3bet';
        case 'facing_4bet': return 'facing_4bet';
        case 'facing_allin_preflop': return 'facing_allin_preflop';
        default: break;
      }
    }

    // Fallback: recompute from raw fields
    const isBB = position === 'BB';
    const isRealRaise = ctx.isRealRaise !== undefined
      ? ctx.isRealRaise
      : currentBet > bb;
    const numLimpers = ctx.numLimpers !== undefined ? ctx.numLimpers : 0;

    if (ctx.facingAllin) return 'facing_allin_preflop';
    if (isBB && !isRealRaise && toCall === 0) return 'bb_check_option';
    if (!isRealRaise) {
      if (numLimpers > 0 && toCall > 0) return 'facing_limp';
      if (numLimpers > 0 && toCall === 0) return 'bb_check_option';
      return 'unopened';
    }
    const raiseLevel = currentBet / bb;
    if (raiseLevel >= 4.0) return 'facing_4bet';
    if (raiseLevel >= 2.3) return 'facing_3bet';
    return 'facing_raise';
  }

  // Apply statistic modifier to a base threshold.
  // Returns adjusted value plus diagnostics about the usage.
  // multiplier: how strongly a strong read influences the threshold (default 0.15 = 15%)
  function applyStatModifier(baseValue, stat, direction, multiplier) {
    const m = (multiplier !== undefined) ? multiplier : 0.15;
    const prior = stat.value;
    const conf = stat.confidence;
    let effect = 0;
    let level = 'ignored';

    if (conf >= STAT_CONFIDENCE.IGNORE) {
      // Determine direction of deviation from prior
      // direction: 'call_more' means higher stat value → loosen threshold (lower value)
      //            'raise_more' means higher stat value → tighten threshold (higher value)
      const delta = (prior - 0.5);
      let rawEffect;
      if (direction === 'fold_to_aggr') {
        // Higher fold stat → opponent folds more → increase aggression (lower call threshold / higher open freq)
        rawEffect = -delta;
      } else if (direction === 'call_station') {
        // Higher call stat → opponent calls more → tighten value (higher threshold)
        rawEffect = delta;
      } else if (direction === 'respect_aggr') {
        // Higher aggr stat → opponent raises more → tighten call threshold
        rawEffect = delta;
      } else {
        rawEffect = delta;
      }

      if (conf < STAT_CONFIDENCE.MICRO) {
        effect = rawEffect * 0.05;
        level = 'micro';
      } else if (conf < STAT_CONFIDENCE.MODERATE) {
        effect = rawEffect * m;
        level = 'moderate';
      } else {
        effect = rawEffect * Math.min(0.30, m * 1.3);
        level = 'strong';
      }
    }

    const adjusted = Math.max(0.05, Math.min(0.95, baseValue + effect));
    return { value: adjusted, effect: effect, level: level };
  }

  // ========================================================================
  // BLOCK 4 — PRO_GTO: The Baseline (Balanced, GTO-approximating)
  // ========================================================================

  const PRO_GTO = {
    key: 'PRO_GTO',
    name: 'GTO Baseline',

    decidePreflop: function(ctx) {
      const profile = PROFILES.PRO_GTO;
      const card1 = ctx.card1, card2 = ctx.card2;
      const position = ctx.position;
      const myStack = ctx.myStack;
      const bb = ctx.bb;
      const toCall = ctx.toCall || 0;
      const pot = ctx.pot || 0;
      const stackBB = myStack / bb;
      const stackDepth = getStackDepth(stackBB);
      const rawStrength = preflopStrength(card1, card2);
      const posMult = getPositionMultiplier(position);
      const blockerScore = getBlockerScore(card1, card2);
      const hasAnte = (ctx.ante > 0) || (ctx.blinds && ctx.blinds.ante > 0);
      const er = equityRealization(card1, card2, position, ctx.numActive || 8);
      const strength = Math.min(1, rawStrength * er);

      const playersLeft = ctx.playersLeft || 8;
      const itmCount = ctx.itmCount || 3;
      const bubbleFactor = calculateBubbleFactor(stackBB, ctx.opponentStackBB || stackBB, playersLeft, itmCount);
      const icmPressure = getICMPressure(playersLeft, itmCount);

      const context = detectContext(ctx);

      const diagnostics = {
        source: 'pro', botType: 'PRO_GTO',
        strength: strength, strengthRaw: rawStrength, equityRealization: er,
        stackDepth, position, positionModifier: posMult, effectiveStackBB: stackBB,
        bubbleFactor, icmPressure, blockerScore,
        multiwayCount: ctx.numActive || 8,
        contextType: context,
        raiserPosition: ctx.raiserPosition || null,
        numLimpers: ctx.numLimpers || 0,
        isRealRaise: !!ctx.isRealRaise,
        opponentType: 'unknown', exploitActive: false,
        statUsed: null, statConfidence: 0, statEffect: 0,
        frequency: 1.0, hasAnte, reason: ''
      };

      // ===== SHORT STACK PUSH/FOLD (only unopened) =====
      if (stackBB <= 20 && context === 'unopened') {
        const table = NASH_PUSH_TABLE[position] || NASH_PUSH_TABLE.BTN;
        let pushFreq = 0.10;
        for (const stack of [5, 8, 10, 12, 15, 20, 25]) {
          if (stackBB <= stack) { pushFreq = table[stack] || pushFreq; break; }
        }
        if (hasAnte) pushFreq *= 1.08;
        if (icmPressure === 'bubble' && stackBB < 15) pushFreq *= 1.15;
        if (icmPressure === 'final_table') pushFreq *= 1.05;
        const roll = Math.random();
        diagnostics.frequency = pushFreq;
        diagnostics.randomRoll = roll;
        diagnostics.threshold = 'nash_push';
        diagnostics.reason = `Nash push ${position} ${stackBB.toFixed(0)}bb freq=${pushFreq.toFixed(2)}`;
        if (roll < pushFreq && strength > 0.30) return { action: 'allin', amount: myStack, diagnostics };
        if (strength < 0.15) return { action: 'fold', amount: 0, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING ALL-IN =====
      if (context === 'facing_allin_preflop') {
        const requiredEquity = bubbleFactor / (bubbleFactor + 1);
        const callThreshold = adjustThresholdForICM(0.55, bubbleFactor, profile.icmSensitivity);
        diagnostics.requiredEquity = requiredEquity;
        diagnostics.threshold = callThreshold;
        diagnostics.reason = `facing allin need ${(requiredEquity * 100).toFixed(0)}% BF=${bubbleFactor.toFixed(2)}`;
        if (strength > callThreshold || (strength > 0.70 && stackBB < 15)) return { action: 'call', amount: toCall, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== BB CHECK OPTION =====
      if (context === 'bb_check_option') {
        // BB has no call to make. If no real raise, check and see the flop.
        // Only consider raising if hand is premium and there were multiple limpers.
        const limpers = ctx.numLimpers || 0;
        if (strength > 0.80 && limpers >= 1) {
          diagnostics.threshold = 0.80;
          diagnostics.reason = `BB iso-raise premium over ${limpers} limper(s)`;
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (hasAnte ? 4.0 : 4.5) + limpers * bb)), diagnostics };
        }
        diagnostics.threshold = 'check';
        diagnostics.reason = 'BB check option, no real raise';
        return { action: 'check', amount: 0, diagnostics };
      }

      // ===== FACING LIMP =====
      if (context === 'facing_limp') {
        const limpers = ctx.numLimpers || 1;
        // Isolation thresholds — raise size scales with limpers
        let isoThreshold = 0.42;
        if (position === 'SB' || position === 'BB') isoThreshold += 0.06;
        if (position === 'BTN' || position === 'CO') isoThreshold -= 0.04;
        if (hasAnte) isoThreshold -= 0.02;
        if (limpers >= 2) isoThreshold -= 0.03;
        isoThreshold = Math.max(0.25, Math.min(0.75, isoThreshold));

        diagnostics.threshold = isoThreshold;
        diagnostics.reason = `iso-raise vs ${limpers} limper(s) thr=${isoThreshold.toFixed(2)}`;

        if (strength > isoThreshold) {
          const isoSize = Math.round(bb * (2.5 + 0.7 * limpers) + (hasAnte ? bb : 0));
          return { action: 'raise', amount: Math.min(myStack, isoSize), diagnostics };
        }
        // Complete small blind or check option calls are handled elsewhere; here just fold
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING RAISE (2-bet) =====
      if (context === 'facing_raise') {
        if (position === 'BB') {
          const openPos = ctx.raiserPosition || 'BTN';
          let defendThreshold = 0.32;
          if (openPos === 'UTG' || openPos === 'UTG1') defendThreshold = 0.48;
          else if (openPos === 'LJ' || openPos === 'HJ') defendThreshold = 0.42;
          else if (openPos === 'CO') defendThreshold = 0.34;
          else defendThreshold = 0.30;
          if (hasAnte) defendThreshold -= 0.04;
          if (icmPressure === 'bubble' && stackBB > 15 && stackBB < 35) defendThreshold += 0.08;
          diagnostics.threshold = defendThreshold;
          diagnostics.reason = `BB defend vs ${openPos} thr=${defendThreshold.toFixed(2)}`;
          if (strength > 0.72 && Math.random() < 0.45) {
            diagnostics.reason += ' | 3bet';
            return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 4.2)), diagnostics };
          }
          if (strength > defendThreshold) return { action: 'call', amount: toCall, diagnostics };
          return { action: 'fold', amount: 0, diagnostics };
        }

        if (position === 'SB') {
          let threeBetThreshold = 0.68;
          if (icmPressure === 'bubble') threeBetThreshold += 0.08;
          diagnostics.threshold = threeBetThreshold;
          diagnostics.reason = `SB vs open 3bet thr=${threeBetThreshold.toFixed(2)}`;
          if (strength > threeBetThreshold) return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 3.6)), diagnostics };
          if (blockerScore > 0.10 && Math.random() < 0.20) {
            diagnostics.reason = 'SB blocker bluff 3bet';
            return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 3.6)), diagnostics };
          }
          return { action: 'fold', amount: 0, diagnostics };
        }

        // Other positions: call or fold based on pot odds and ICM
        const potOdds = toCall / (pot + toCall);
        const callThreshold = Math.max(0.42, potOdds * 1.5);
        diagnostics.threshold = callThreshold;
        diagnostics.potOdds = potOdds;
        diagnostics.reason = `facing raise potOdds=${(potOdds * 100).toFixed(0)}%`;
        if (strength > callThreshold) return { action: 'call', amount: toCall, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING 3-BET =====
      if (context === 'facing_3bet') {
        // 4-bet or fold (with some flatting in position)
        const fourBetThreshold = 0.80;
        const flatThreshold = 0.55;
        diagnostics.threshold = fourBetThreshold;
        diagnostics.reason = 'facing 3bet: 4bet or fold/flat';

        if (strength > fourBetThreshold && stackBB > 15) {
          diagnostics.reason += ' | 4bet';
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 9)), diagnostics };
        }
        if ((position === 'BTN' || position === 'CO') && strength > flatThreshold && stackBB > 25) {
          diagnostics.reason += ' | in-position flat';
          return { action: 'call', amount: toCall, diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING 4-BET =====
      if (context === 'facing_4bet') {
        // Very tight — only AA, KK, and sometimes QQ/AKs
        const fiveBetThreshold = 0.88;
        diagnostics.threshold = fiveBetThreshold;
        diagnostics.reason = `facing 4bet thr=${fiveBetThreshold.toFixed(2)}`;
        if (strength > fiveBetThreshold && stackBB > 20) {
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 22)), diagnostics };
        }
        if (strength > 0.78 && stackBB < 20) return { action: 'allin', amount: myStack, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== UNOPENED (RFI) =====
      const baseThreshold = BASE_OPEN_THRESHOLDS[position] || 0.50;
      let openThreshold = baseThreshold / posMult;
      if (hasAnte) openThreshold *= 0.92;
      if (icmPressure === 'bubble') {
        if (stackBB > 30) openThreshold *= 0.90;
        else if (stackBB > 15) openThreshold *= 1.12;
      }
      if (icmPressure === 'final_table') {
        if (stackBB > 30) openThreshold *= 0.88;
        else if (stackBB > 15) openThreshold *= 1.15;
      }
      openThreshold = Math.min(0.95, Math.max(0.10, openThreshold));

      if (position === 'SB') {
        const limpFreq = profile.limpHabit * (hasAnte ? 1.3 : 1.0);
        const roll = Math.random();
        diagnostics.frequency = 1.0; diagnostics.limpFreq = limpFreq; diagnostics.threshold = openThreshold;
        diagnostics.reason = `SB RFI thr=${openThreshold.toFixed(2)} limp=${limpFreq.toFixed(2)}`;
        if (strength > openThreshold) {
          if (roll < limpFreq && strength > 0.75) {
            diagnostics.reason = 'SB balance limp';
            return { action: 'call', amount: bb, diagnostics };
          }
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (hasAnte ? 2.2 : 2.5))), diagnostics };
        }
        if (strength > 0.35 && roll < limpFreq * 1.5) {
          diagnostics.reason = 'SB speculative limp';
          return { action: 'call', amount: bb, diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      diagnostics.threshold = openThreshold;
      diagnostics.reason = `RFI ${position} thr=${openThreshold.toFixed(2)}`;
      if (strength > openThreshold) {
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (hasAnte ? 2.0 : 2.5))), diagnostics };
      }
      return { action: 'fold', amount: 0, diagnostics };
    }
  };

  // ========================================================================
  // BLOCK 5 — PRO_SOLVER: Aggressive, Blocker-Aware, Uses Shrunk Stats
  // ========================================================================

  const PRO_SOLVER = {
    key: 'PRO_SOLVER',
    name: 'Solver Aggressive',

    decidePreflop: function(ctx) {
      const profile = PROFILES.PRO_SOLVER;
      const card1 = ctx.card1, card2 = ctx.card2;
      const position = ctx.position;
      const myStack = ctx.myStack;
      const bb = ctx.bb;
      const toCall = ctx.toCall || 0;
      const pot = ctx.pot || 0;
      const stackBB = myStack / bb;
      const stackDepth = getStackDepth(stackBB);
      const rawStrength = preflopStrength(card1, card2);
      const posMult = getPositionMultiplier(position);
      const blockerScore = getBlockerScore(card1, card2);
      const hasAnte = (ctx.ante > 0) || (ctx.blinds && ctx.blinds.ante > 0);
      const er = equityRealization(card1, card2, position, ctx.numActive || 8);
      const strength = Math.min(1, rawStrength * er);

      const playersLeft = ctx.playersLeft || 8;
      const itmCount = ctx.itmCount || 3;
      const bubbleFactor = calculateBubbleFactor(stackBB, ctx.opponentStackBB || stackBB, playersLeft, itmCount);
      const icmPressure = getICMPressure(playersLeft, itmCount);
      const context = detectContext(ctx);

      const oppId = ctx.opponentId || null;
      const oppType = oppId ? getOpponentType(oppId) : 'unknown';
      const foldToResteal = oppId ? getShrunk(oppId, 'foldToResteal') : { value: PRIORS.foldToResteal, confidence: 0 };
      const foldTo4Bet = oppId ? getShrunk(oppId, 'foldTo4Bet') : { value: PRIORS.foldTo4Bet, confidence: 0 };
      const foldToSteal = oppId ? getShrunk(oppId, 'foldBbToStealBtn') : { value: PRIORS.foldBbToStealBtn, confidence: 0 };

      const diagnostics = {
        source: 'pro', botType: 'PRO_SOLVER',
        strength, strengthRaw: rawStrength, equityRealization: er,
        stackDepth, position, positionModifier: posMult, effectiveStackBB: stackBB,
        bubbleFactor, icmPressure, blockerScore,
        multiwayCount: ctx.numActive || 8,
        contextType: context,
        raiserPosition: ctx.raiserPosition || null,
        numLimpers: ctx.numLimpers || 0,
        isRealRaise: !!ctx.isRealRaise,
        opponentType: oppType, exploitActive: false,
        statUsed: null, statConfidence: 0, statEffect: 0,
        hasAnte, frequency: 1.0, reason: ''
      };

      // ===== SHORT STACK PUSH/FOLD =====
      if (stackBB <= 20 && context === 'unopened') {
        const table = NASH_PUSH_TABLE[position] || NASH_PUSH_TABLE.BTN;
        let pushFreq = 0.10;
        for (const stack of [5, 8, 10, 12, 15, 20, 25]) {
          if (stackBB <= stack) { pushFreq = table[stack] || pushFreq; break; }
        }
        pushFreq *= 1.08;
        if (hasAnte) pushFreq *= 1.06;
        if (icmPressure === 'bubble' && stackBB < 15) pushFreq *= 1.10;
        const roll = Math.random();
        diagnostics.frequency = pushFreq; diagnostics.randomRoll = roll;
        diagnostics.threshold = 'nash_push_aggressive';
        diagnostics.reason = `Nash push solver freq=${pushFreq.toFixed(2)}`;
        if (roll < pushFreq && strength > 0.28) return { action: 'allin', amount: myStack, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING ALL-IN =====
      if (context === 'facing_allin_preflop') {
        const callThreshold = adjustThresholdForICM(0.52, bubbleFactor, profile.icmSensitivity);
        diagnostics.threshold = callThreshold;
        diagnostics.reason = `facing allin BF=${bubbleFactor.toFixed(2)}`;
        if (strength > callThreshold || (blockerScore > 0.10 && strength > 0.45 && stackBB < 20))
          return { action: 'call', amount: toCall, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== BB CHECK OPTION =====
      if (context === 'bb_check_option') {
        const limpers = ctx.numLimpers || 0;
        // Aggressive iso-raise with strong hands over limpers
        if (strength > 0.68 && limpers >= 1) {
          diagnostics.threshold = 0.68;
          diagnostics.reason = `BB iso-raise over ${limpers} limper(s)`;
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (3.5 + limpers))), diagnostics };
        }
        diagnostics.threshold = 'check';
        diagnostics.reason = 'BB check option, no raise';
        return { action: 'check', amount: 0, diagnostics };
      }

      // ===== FACING LIMP =====
      if (context === 'facing_limp') {
        const limpers = ctx.numLimpers || 1;
        let isoThreshold = 0.36;
        if (position === 'SB' || position === 'BB') isoThreshold += 0.05;
        if (hasAnte) isoThreshold -= 0.02;
        if (limpers >= 2) isoThreshold -= 0.04;
        isoThreshold = Math.max(0.22, Math.min(0.70, isoThreshold));

        diagnostics.threshold = isoThreshold;
        diagnostics.reason = `solver iso-raise vs ${limpers} limper(s) thr=${isoThreshold.toFixed(2)}`;

        if (strength > isoThreshold) {
          const isoSize = Math.round(bb * (3.0 + 0.8 * limpers) + (hasAnte ? bb : 0));
          return { action: 'raise', amount: Math.min(myStack, isoSize), diagnostics };
        }
        if (strength > 0.30 && blockerScore > 0.05) {
          diagnostics.reason += ' | blocker overcall';
          return { action: 'call', amount: toCall, diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING RAISE =====
      if (context === 'facing_raise') {
        if (position === 'BB') {
          const openPos = ctx.raiserPosition || 'BTN';
          let defendThreshold = 0.30;
          if (openPos === 'UTG' || openPos === 'UTG1') defendThreshold = 0.46;
          else if (openPos === 'LJ' || openPos === 'HJ') defendThreshold = 0.40;
          else if (openPos === 'CO') defendThreshold = 0.32;
          else defendThreshold = 0.28;

          // Stat-based adjustment: over-folding opponent → 3bet wider
          let statAdj = { value: defendThreshold, effect: 0, level: 'ignored' };
          let threeBetWider = false;
          if (foldToResteal.confidence >= STAT_CONFIDENCE.MICRO && foldToResteal.value > 0.58) {
            statAdj = applyStatModifier(defendThreshold, foldToResteal, 'fold_to_aggr', 0.15);
            threeBetWider = true;
            diagnostics.statUsed = 'foldToResteal';
            diagnostics.statConfidence = foldToResteal.confidence;
            diagnostics.statEffect = statAdj.effect;
            diagnostics.exploitActive = statAdj.level !== 'ignored';
          }
          defendThreshold = statAdj.value;

          if (strength > 0.65 || (blockerScore > 0.08 && Math.random() < (threeBetWider ? 0.55 : 0.32))) {
            diagnostics.threshold = 0.65;
            diagnostics.reason = `BB 3bet solver blocker=${blockerScore.toFixed(2)} exploitable=${threeBetWider}`;
            return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 4.2)), diagnostics };
          }
          diagnostics.threshold = defendThreshold;
          diagnostics.reason = `BB defend vs ${openPos} solver thr=${defendThreshold.toFixed(2)}`;
          if (strength > defendThreshold) return { action: 'call', amount: toCall, diagnostics };
          return { action: 'fold', amount: 0, diagnostics };
        }

        if (position === 'SB') {
          let threeBetThreshold = 0.60;
          diagnostics.threshold = threeBetThreshold;
          diagnostics.reason = `SB solver 3bet thr=${threeBetThreshold.toFixed(2)}`;
          if (strength > threeBetThreshold || (blockerScore > 0.10 && Math.random() < 0.35)) {
            return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 3.4)), diagnostics };
          }
          return { action: 'fold', amount: 0, diagnostics };
        }

        // Other positions — stat-aware call threshold
        let callThreshold = 0.42;
        if (oppType === 'nit' && oppId) {
          const nitStat = { value: 0.65, confidence: 0.7 };
          const adj = applyStatModifier(callThreshold, nitStat, 'respect_aggr', 0.15);
          callThreshold = adj.value;
          diagnostics.statUsed = 'opponentType';
          diagnostics.statConfidence = 0.7;
          diagnostics.statEffect = adj.effect;
        } else if (oppType === 'maniac') {
          const maniacStat = { value: 0.30, confidence: 0.7 };
          const adj = applyStatModifier(callThreshold, maniacStat, 'respect_aggr', 0.15);
          callThreshold = adj.value;
          diagnostics.statUsed = 'opponentType';
          diagnostics.statConfidence = 0.7;
          diagnostics.statEffect = adj.effect;
        }
        diagnostics.threshold = callThreshold;
        diagnostics.reason = diagnostics.reason || `facing raise solver opp=${oppType}`;
        if (strength > callThreshold) return { action: 'call', amount: toCall, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING 3-BET =====
      if (context === 'facing_3bet') {
        const fourBetThreshold = 0.78;
        let fourBetFreq = 0.65;
        if (foldTo4Bet.confidence >= STAT_CONFIDENCE.MICRO) {
          if (foldTo4Bet.value > 0.60) {
            fourBetFreq = 0.85;
            diagnostics.statUsed = 'foldTo4Bet';
            diagnostics.statConfidence = foldTo4Bet.confidence;
            diagnostics.statEffect = 0.20;
            diagnostics.exploitActive = true;
          } else if (foldTo4Bet.value < 0.30) {
            fourBetFreq = 0.35;
            diagnostics.statUsed = 'foldTo4Bet';
            diagnostics.statConfidence = foldTo4Bet.confidence;
            diagnostics.statEffect = -0.30;
            diagnostics.exploitActive = true;
          }
        }
        diagnostics.threshold = fourBetThreshold;
        diagnostics.reason = `solver vs 3bet fourBetFreq=${fourBetFreq.toFixed(2)}`;

        if (strength > fourBetThreshold && Math.random() < fourBetFreq && stackBB > 15) {
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 9)), diagnostics };
        }
        if ((position === 'BTN' || position === 'CO') && strength > 0.55 && stackBB > 25) {
          diagnostics.reason += ' | flat IP';
          return { action: 'call', amount: toCall, diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING 4-BET =====
      if (context === 'facing_4bet') {
        let fiveBetThreshold = 0.86;
        if (foldTo4Bet.confidence >= STAT_CONFIDENCE.MICRO) {
          if (foldTo4Bet.value > 0.65) {
            fiveBetThreshold = 0.78;
            diagnostics.statUsed = 'foldTo4Bet';
            diagnostics.statConfidence = foldTo4Bet.confidence;
            diagnostics.statEffect = -0.08;
            diagnostics.exploitActive = true;
          } else if (foldTo4Bet.value < 0.30) {
            fiveBetThreshold = 0.92;
            diagnostics.statUsed = 'foldTo4Bet';
            diagnostics.statConfidence = foldTo4Bet.confidence;
            diagnostics.statEffect = 0.06;
            diagnostics.exploitActive = true;
          }
        }
        diagnostics.threshold = fiveBetThreshold;
        diagnostics.reason = `solver facing 4bet thr=${fiveBetThreshold.toFixed(2)}`;
        if (strength > fiveBetThreshold && stackBB > 20) {
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 22)), diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== UNOPENED (RFI) =====
      const baseThreshold = BASE_OPEN_THRESHOLDS[position] || 0.50;
      let openThreshold = (baseThreshold / posMult) * 0.92;
      if (hasAnte) openThreshold *= 0.90;

      // Stat-based adjustment to open threshold
      if (oppType === 'station' && oppId) {
        const adj = applyStatModifier(openThreshold, { value: 0.75, confidence: 0.65 }, 'call_station', 0.15);
        openThreshold = adj.value;
        diagnostics.statUsed = 'opponentType';
        diagnostics.statConfidence = 0.65;
        diagnostics.statEffect = adj.effect;
      } else if (oppType === 'nit' && oppId) {
        const adj = applyStatModifier(openThreshold, { value: 0.25, confidence: 0.65 }, 'fold_to_aggr', 0.15);
        openThreshold = adj.value;
        diagnostics.statUsed = 'opponentType';
        diagnostics.statConfidence = 0.65;
        diagnostics.statEffect = adj.effect;
      }
      openThreshold = Math.min(0.90, Math.max(0.15, openThreshold));

      if (position === 'SB') {
        const limpFreq = profile.limpHabit * (hasAnte ? 1.2 : 1.0);
        const roll = Math.random();
        diagnostics.threshold = openThreshold; diagnostics.limpFreq = limpFreq;
        diagnostics.reason = `SB RFI solver thr=${openThreshold.toFixed(2)}`;
        if (strength > openThreshold) {
          if (roll < limpFreq * 0.8 && strength > 0.78) {
            diagnostics.reason = 'SB balance limp';
            return { action: 'call', amount: bb, diagnostics };
          }
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (hasAnte ? 2.0 : 2.4))), diagnostics };
        }
        if (strength > 0.32 && roll < limpFreq) return { action: 'call', amount: bb, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      diagnostics.threshold = openThreshold;
      diagnostics.reason = `RFI ${position} solver thr=${openThreshold.toFixed(2)}`;
      if (strength > openThreshold) {
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (hasAnte ? 1.9 : 2.3))), diagnostics };
      }
      return { action: 'fold', amount: 0, diagnostics };
    }
  };

  // ========================================================================
  // BLOCK 6 — PRO_ICM: ICM-Maximal, Bubble Specialist
  // ========================================================================

  const PRO_ICM = {
    key: 'PRO_ICM',
    name: 'ICM Specialist',

    decidePreflop: function(ctx) {
      const profile = PROFILES.PRO_ICM;
      const card1 = ctx.card1, card2 = ctx.card2;
      const position = ctx.position;
      const myStack = ctx.myStack;
      const bb = ctx.bb;
      const toCall = ctx.toCall || 0;
      const pot = ctx.pot || 0;
      const stackBB = myStack / bb;
      const stackDepth = getStackDepth(stackBB);
      const rawStrength = preflopStrength(card1, card2);
      const posMult = getPositionMultiplier(position);
      const blockerScore = getBlockerScore(card1, card2);
      const hasAnte = (ctx.ante > 0) || (ctx.blinds && ctx.blinds.ante > 0);
      const er = equityRealization(card1, card2, position, ctx.numActive || 8);
      const strength = Math.min(1, rawStrength * er);

      const playersLeft = ctx.playersLeft || 8;
      const itmCount = ctx.itmCount || 3;
      const oppStackBB = ctx.opponentStackBB || stackBB;
      const bubbleFactor = calculateBubbleFactor(stackBB, oppStackBB, playersLeft, itmCount);
      const icmPressure = getICMPressure(playersLeft, itmCount);
      const payoutStructure = getPayoutStructure(playersLeft, itmCount);
      const context = detectContext(ctx);

      const oppId = ctx.opponentId || null;
      const oppType = oppId ? getOpponentType(oppId) : 'unknown';
      const foldToResteal = oppId ? getShrunk(oppId, 'foldToResteal') : { value: PRIORS.foldToResteal, confidence: 0 };
      const isMediumOnBubble = icmPressure === 'bubble' && stackBB > 15 && stackBB < 35;
      const isBigOnBubble = icmPressure === 'bubble' && stackBB >= 35;

      const diagnostics = {
        source: 'pro', botType: 'PRO_ICM',
        strength, strengthRaw: rawStrength, equityRealization: er,
        stackDepth, position, positionModifier: posMult, effectiveStackBB: stackBB,
        bubbleFactor, icmPressure, payoutStructure, blockerScore,
        multiwayCount: ctx.numActive || 8,
        contextType: context,
        raiserPosition: ctx.raiserPosition || null,
        numLimpers: ctx.numLimpers || 0,
        isRealRaise: !!ctx.isRealRaise,
        opponentType: oppType, exploitActive: false,
        statUsed: null, statConfidence: 0, statEffect: 0,
        hasAnte, frequency: 1.0, reason: ''
      };

      // ===== SHORT STACK PUSH =====
      if (stackBB <= 20 && context === 'unopened') {
        const table = NASH_PUSH_TABLE[position] || NASH_PUSH_TABLE.BTN;
        let pushFreq = 0.10;
        for (const stack of [5, 8, 10, 12, 15, 20, 25]) {
          if (stackBB <= stack) { pushFreq = table[stack] || pushFreq; break; }
        }
        if (hasAnte) pushFreq *= 1.06;
        if (icmPressure === 'bubble' && stackBB < 15) pushFreq *= 1.20;
        if (icmPressure === 'final_table' && stackBB < 15) pushFreq *= 1.10;
        if (icmPressure === 'bubble' && stackBB >= 15 && stackBB <= 30) pushFreq *= 0.75;
        if (icmPressure === 'final_table' && stackBB >= 15 && stackBB <= 30) pushFreq *= 0.80;
        const roll = Math.random();
        diagnostics.frequency = pushFreq; diagnostics.randomRoll = roll;
        diagnostics.threshold = 'nash_push_icm';
        diagnostics.reason = `Nash push ICM ${icmPressure} freq=${pushFreq.toFixed(2)}`;
        if (roll < pushFreq && strength > 0.28) return { action: 'allin', amount: myStack, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING ALL-IN =====
      if (context === 'facing_allin_preflop') {
        const requiredEquity = bubbleFactor / (bubbleFactor + 1);
        let callThreshold = adjustThresholdForICM(0.55, bubbleFactor, profile.icmSensitivity);
        if (isMediumOnBubble) callThreshold += 0.12;
        if (icmPressure === 'final_table' && stackBB > 15 && stackBB < 35) callThreshold += 0.10;
        diagnostics.threshold = callThreshold;
        diagnostics.requiredEquity = requiredEquity;
        diagnostics.reason = `facing allin ICM need ${(callThreshold * 100).toFixed(0)}% BF=${bubbleFactor.toFixed(2)}`;
        if (strength > callThreshold || (stackBB < 10 && strength > 0.60)) return { action: 'call', amount: toCall, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== BB CHECK OPTION =====
      if (context === 'bb_check_option') {
        const limpers = ctx.numLimpers || 0;
        // ICM: on bubble, prefer checking. Only iso-raise premium hands.
        const isoThreshold = (isMediumOnBubble || isBigOnBubble) ? 0.88 : 0.82;
        if (strength > isoThreshold && limpers >= 1) {
          diagnostics.threshold = isoThreshold;
          diagnostics.reason = `BB ICM iso-raise over ${limpers} limper(s) thr=${isoThreshold.toFixed(2)}`;
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (4.0 + limpers))), diagnostics };
        }
        diagnostics.threshold = 'check';
        diagnostics.reason = 'BB check option, ICM cautious';
        return { action: 'check', amount: 0, diagnostics };
      }

      // ===== FACING LIMP =====
      if (context === 'facing_limp') {
        const limpers = ctx.numLimpers || 1;
        let isoThreshold = 0.48;
        if (position === 'SB' || position === 'BB') isoThreshold += 0.06;
        if (hasAnte) isoThreshold -= 0.02;
        if (isMediumOnBubble) isoThreshold += 0.08;
        if (isBigOnBubble) isoThreshold -= 0.04;
        isoThreshold = Math.max(0.30, Math.min(0.80, isoThreshold));

        diagnostics.threshold = isoThreshold;
        diagnostics.reason = `ICM iso-raise vs ${limpers} limper(s) thr=${isoThreshold.toFixed(2)}`;

        if (strength > isoThreshold) {
          const isoSize = Math.round(bb * (2.5 + 0.7 * limpers) + (hasAnte ? bb : 0));
          return { action: 'raise', amount: Math.min(myStack, isoSize), diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING RAISE =====
      if (context === 'facing_raise') {
        if (position === 'BB') {
          const openPos = ctx.raiserPosition || 'BTN';
          let defendThreshold = 0.34;
          if (openPos === 'UTG' || openPos === 'UTG1') defendThreshold = 0.50;
          else if (openPos === 'LJ' || openPos === 'HJ') defendThreshold = 0.44;
          else if (openPos === 'CO') defendThreshold = 0.36;
          else defendThreshold = 0.32;
          if (isMediumOnBubble) defendThreshold += 0.10;
          if (icmPressure === 'final_table' && stackBB > 15 && stackBB < 35) defendThreshold += 0.08;
          diagnostics.threshold = defendThreshold;
          diagnostics.reason = `BB defend ICM vs ${openPos} thr=${defendThreshold.toFixed(2)}`;
          if (strength > defendThreshold) return { action: 'call', amount: toCall, diagnostics };
          return { action: 'fold', amount: 0, diagnostics };
        }

        if (position === 'SB') {
          let threeBetThreshold = 0.72;
          if (icmPressure === 'bubble') threeBetThreshold += 0.08;
          diagnostics.threshold = threeBetThreshold;
          diagnostics.reason = `SB 3bet ICM thr=${threeBetThreshold.toFixed(2)}`;
          if (strength > threeBetThreshold) return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 3.4)), diagnostics };
          return { action: 'fold', amount: 0, diagnostics };
        }

        const potOdds = toCall / (pot + toCall);
        let callThreshold = Math.max(0.44, potOdds * 1.6);
        if (isMediumOnBubble) callThreshold += 0.10;
        diagnostics.threshold = callThreshold;
        diagnostics.potOdds = potOdds;
        diagnostics.reason = `facing raise ICM potOdds=${(potOdds * 100).toFixed(0)}%`;
        if (strength > callThreshold) return { action: 'call', amount: toCall, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING 3-BET =====
      if (context === 'facing_3bet') {
        const fourBetThreshold = 0.84;
        diagnostics.threshold = fourBetThreshold;
        diagnostics.reason = `ICM vs 3bet thr=${fourBetThreshold.toFixed(2)}`;
        if (strength > fourBetThreshold && stackBB > 20) {
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 9)), diagnostics };
        }
        if ((position === 'BTN' || position === 'CO') && strength > 0.60 && stackBB > 30) {
          diagnostics.reason += ' | in-position flat';
          return { action: 'call', amount: toCall, diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING 4-BET =====
      if (context === 'facing_4bet') {
        const fiveBetThreshold = 0.92;
        diagnostics.threshold = fiveBetThreshold;
        diagnostics.reason = `ICM vs 4bet thr=${fiveBetThreshold.toFixed(2)}`;
        if (strength > fiveBetThreshold && stackBB > 25) {
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 22)), diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== UNOPENED (RFI) =====
      const baseThreshold = BASE_OPEN_THRESHOLDS[position] || 0.50;
      let openThreshold = baseThreshold / posMult;
      if (hasAnte) openThreshold *= 0.94;
      if (isBigOnBubble) openThreshold *= 0.85;
      else if (isMediumOnBubble) openThreshold *= 1.15;
      if (icmPressure === 'final_table') {
        if (stackBB > 40) openThreshold *= 0.82;
        else if (stackBB > 15 && stackBB <= 40) openThreshold *= 1.18;
      }
      openThreshold = Math.min(0.92, Math.max(0.15, openThreshold));

      if (position === 'SB') {
        const limpFreq = profile.limpHabit * (hasAnte ? 1.2 : 1.0) * (icmPressure === 'bubble' ? 0.5 : 1.0);
        const roll = Math.random();
        diagnostics.threshold = openThreshold; diagnostics.limpFreq = limpFreq;
        diagnostics.reason = `SB RFI ICM thr=${openThreshold.toFixed(2)} limp=${limpFreq.toFixed(2)}`;
        if (strength > openThreshold) {
          if (roll < limpFreq && strength > 0.80) {
            diagnostics.reason = 'SB balance limp ICM';
            return { action: 'call', amount: bb, diagnostics };
          }
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (hasAnte ? 2.1 : 2.5))), diagnostics };
        }
        if (strength > 0.35 && roll < limpFreq) return { action: 'call', amount: bb, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      diagnostics.threshold = openThreshold;
      diagnostics.reason = `RFI ${position} ICM thr=${openThreshold.toFixed(2)}`;
      if (strength > openThreshold) {
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (hasAnte ? 2.0 : 2.5))), diagnostics };
      }
      return { action: 'fold', amount: 0, diagnostics };
    }
  };

  // ========================================================================
  // BLOCK 7 — PRO_LAG: Maximum Aggression, Exploit-Heavy
  // ========================================================================

  const PRO_LAG = {
    key: 'PRO_LAG',
    name: 'LAG Aggressor',

    decidePreflop: function(ctx) {
      const profile = PROFILES.PRO_LAG;
      const card1 = ctx.card1, card2 = ctx.card2;
      const position = ctx.position;
      const myStack = ctx.myStack;
      const bb = ctx.bb;
      const toCall = ctx.toCall || 0;
      const pot = ctx.pot || 0;
      const stackBB = myStack / bb;
      const stackDepth = getStackDepth(stackBB);
      const rawStrength = preflopStrength(card1, card2);
      const posMult = getPositionMultiplier(position);
      const blockerScore = getBlockerScore(card1, card2);
      const hasAnte = (ctx.ante > 0) || (ctx.blinds && ctx.blinds.ante > 0);
      const er = equityRealization(card1, card2, position, ctx.numActive || 8);
      const strength = Math.min(1, rawStrength * er);

      const playersLeft = ctx.playersLeft || 8;
      const itmCount = ctx.itmCount || 3;
      const bubbleFactor = calculateBubbleFactor(stackBB, ctx.opponentStackBB || stackBB, playersLeft, itmCount);
      const icmPressure = getICMPressure(playersLeft, itmCount);
      const effectiveICM = icmPressure === 'final_table' ? icmPressure : 'none';
      const context = detectContext(ctx);

      const oppId = ctx.opponentId || null;
      const oppType = oppId ? getOpponentType(oppId) : 'unknown';
      const foldToResteal = oppId ? getShrunk(oppId, 'foldToResteal') : { value: PRIORS.foldToResteal, confidence: 0 };
      const foldTo4Bet = oppId ? getShrunk(oppId, 'foldTo4Bet') : { value: PRIORS.foldTo4Bet, confidence: 0 };
      const foldToSqueeze = oppId ? getShrunk(oppId, 'foldToSqueeze') : { value: PRIORS.foldToSqueeze, confidence: 0 };

      const diagnostics = {
        source: 'pro', botType: 'PRO_LAG',
        strength, strengthRaw: rawStrength, equityRealization: er,
        stackDepth, position, positionModifier: posMult, effectiveStackBB: stackBB,
        bubbleFactor, icmPressure, blockerScore,
        multiwayCount: ctx.numActive || 8,
        contextType: context,
        raiserPosition: ctx.raiserPosition || null,
        numLimpers: ctx.numLimpers || 0,
        isRealRaise: !!ctx.isRealRaise,
        opponentType: oppType, exploitActive: false,
        statUsed: null, statConfidence: 0, statEffect: 0,
        hasAnte, frequency: 1.0, reason: ''
      };

      // ===== SHORT STACK =====
      if (stackBB <= 20 && context === 'unopened') {
        const table = NASH_PUSH_TABLE[position] || NASH_PUSH_TABLE.BTN;
        let pushFreq = 0.10;
        for (const stack of [5, 8, 10, 12, 15, 20, 25]) {
          if (stackBB <= stack) { pushFreq = table[stack] || pushFreq; break; }
        }
        pushFreq *= 1.15;
        if (hasAnte) pushFreq *= 1.05;
        if (effectiveICM === 'final_table') pushFreq *= 0.90;
        const roll = Math.random();
        diagnostics.frequency = pushFreq; diagnostics.randomRoll = roll;
        diagnostics.threshold = 'nash_push_lag';
        diagnostics.reason = `Nash push LAG freq=${pushFreq.toFixed(2)}`;
        if (roll < pushFreq && strength > 0.25) return { action: 'allin', amount: myStack, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING ALL-IN =====
      if (context === 'facing_allin_preflop') {
        const callThreshold = Math.max(0.45, adjustThresholdForICM(0.48, bubbleFactor * 0.7, profile.icmSensitivity));
        diagnostics.threshold = callThreshold;
        diagnostics.reason = `facing allin LAG thr=${callThreshold.toFixed(2)}`;
        if (strength > callThreshold) return { action: 'call', amount: toCall, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== BB CHECK OPTION =====
      if (context === 'bb_check_option') {
        const limpers = ctx.numLimpers || 0;
        if (strength > 0.60 && limpers >= 1) {
          diagnostics.threshold = 0.60;
          diagnostics.reason = `BB LAG iso-raise over ${limpers} limper(s)`;
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (3.5 + limpers))), diagnostics };
        }
        // LAG sometimes raises with weak-but-playable hands for balance
        if (limpers >= 1 && Math.random() < 0.20 && blockerScore > 0) {
          diagnostics.reason = 'BB LAG blocker iso';
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (4.0 + limpers))), diagnostics };
        }
        diagnostics.threshold = 'check';
        diagnostics.reason = 'BB check option LAG';
        return { action: 'check', amount: 0, diagnostics };
      }

      // ===== FACING LIMP =====
      if (context === 'facing_limp') {
        const limpers = ctx.numLimpers || 1;
        let isoThreshold = 0.32;
        if (hasAnte) isoThreshold -= 0.02;
        if (limpers >= 2) isoThreshold -= 0.03;
        isoThreshold = Math.max(0.20, Math.min(0.65, isoThreshold));

        diagnostics.threshold = isoThreshold;
        diagnostics.reason = `LAG iso-raise vs ${limpers} limper(s) thr=${isoThreshold.toFixed(2)}`;

        if (strength > isoThreshold) {
          const isoSize = Math.round(bb * (3.2 + 0.9 * limpers) + (hasAnte ? bb : 0));
          return { action: 'raise', amount: Math.min(myStack, isoSize), diagnostics };
        }
        if (strength > 0.28 && blockerScore > 0 && Math.random() < 0.30) {
          diagnostics.reason += ' | LAG overcall bluff';
          return { action: 'call', amount: toCall, diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING RAISE =====
      if (context === 'facing_raise') {
        // Against 4-bet: use foldTo4Bet if enough sample
        if (foldTo4Bet.confidence >= STAT_CONFIDENCE.MICRO) {
          if (foldTo4Bet.value < 0.30) {
            diagnostics.statUsed = 'foldTo4Bet';
            diagnostics.statConfidence = foldTo4Bet.confidence;
            diagnostics.statEffect = 0.10;
            diagnostics.threshold = 0.65;
            diagnostics.reason = 'LAG vs 4bet, opp never folds, only continue strong';
            if (strength > 0.65) return { action: 'call', amount: toCall, diagnostics };
            return { action: 'fold', amount: 0, diagnostics };
          }
          if (foldTo4Bet.value > 0.65 && blockerScore > 0) {
            diagnostics.statUsed = 'foldTo4Bet';
            diagnostics.statConfidence = foldTo4Bet.confidence;
            diagnostics.statEffect = -0.20;
            diagnostics.exploitActive = true;
            diagnostics.reason = 'LAG vs 4bet, opp overfolds, bluff 4bet';
            return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 22)), diagnostics };
          }
        }

        const threeBetFreq = 0.25 + (blockerScore > 0 ? 0.15 : 0);
        const roll = Math.random();

        if (position === 'BB') {
          const openPos = ctx.raiserPosition || 'BTN';
          const rest = foldToResteal.confidence >= STAT_CONFIDENCE.MICRO && foldToResteal.value > 0.60;
          if (rest) {
            diagnostics.statUsed = 'foldToResteal';
            diagnostics.statConfidence = foldToResteal.confidence;
            diagnostics.statEffect = 0.15;
            diagnostics.exploitActive = true;
          }
          if (strength > 0.58 || (blockerScore > 0.05 && roll < (rest ? 0.55 : threeBetFreq))) {
            diagnostics.threshold = 0.58;
            diagnostics.reason = `BB 3bet LAG blocker=${blockerScore.toFixed(2)} exploitable=${rest}`;
            return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 4.5)), diagnostics };
          }
          diagnostics.threshold = 0.28;
          diagnostics.reason = `BB defend LAG vs ${openPos}`;
          if (strength > 0.28) return { action: 'call', amount: toCall, diagnostics };
          return { action: 'fold', amount: 0, diagnostics };
        }

        if (position === 'SB') {
          if (strength > 0.55 || (blockerScore > 0.05 && roll < 0.35)) {
            return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 3.5)), diagnostics };
          }
          return { action: 'fold', amount: 0, diagnostics };
        }

        if (strength > 0.55 && roll < threeBetFreq) {
          diagnostics.reason = 'LAG 3bet';
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 3.6)), diagnostics };
        }

        let callThreshold = 0.38;
        if (oppType === 'nit') callThreshold = 0.50;
        if (oppType === 'maniac') callThreshold = 0.30;
        diagnostics.threshold = callThreshold;
        diagnostics.reason = `facing raise LAG opp=${oppType}`;
        if (strength > callThreshold) return { action: 'call', amount: toCall, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING 3-BET =====
      if (context === 'facing_3bet') {
        const fourBetThreshold = 0.70;
        let fourBetFreq = 0.55;
        if (foldTo4Bet.confidence >= STAT_CONFIDENCE.MODERATE && foldTo4Bet.value > 0.65) {
          fourBetFreq = 0.80;
          diagnostics.statUsed = 'foldTo4Bet';
          diagnostics.statConfidence = foldTo4Bet.confidence;
          diagnostics.statEffect = 0.25;
          diagnostics.exploitActive = true;
        }
        diagnostics.threshold = fourBetThreshold;
        diagnostics.reason = `LAG vs 3bet thr=${fourBetThreshold.toFixed(2)} freq=${fourBetFreq.toFixed(2)}`;
        if (strength > fourBetThreshold && Math.random() < fourBetFreq && stackBB > 15) {
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * 9)), diagnostics };
        }
        if ((position === 'BTN' || position === 'CO') && strength > 0.50 && stackBB > 25) {
          diagnostics.reason += ' | in-position flat';
          return { action: 'call', amount: toCall, diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== FACING 4-BET =====
      if (context === 'facing_4bet') {
        let fiveBetThreshold = 0.82;
        if (foldTo4Bet.confidence >= STAT_CONFIDENCE.MICRO && foldTo4Bet.value > 0.60) {
          fiveBetThreshold = 0.74;
          diagnostics.statUsed = 'foldTo4Bet';
          diagnostics.statConfidence = foldTo4Bet.confidence;
          diagnostics.statEffect = -0.08;
          diagnostics.exploitActive = true;
        }
        diagnostics.threshold = fiveBetThreshold;
        diagnostics.reason = `LAG vs 4bet thr=${fiveBetThreshold.toFixed(2)}`;
        if (strength > fiveBetThreshold && stackBB > 15) {
          return { action: 'allin', amount: myStack, diagnostics };
        }
        return { action: 'fold', amount: 0, diagnostics };
      }

      // ===== UNOPENED (RFI) =====
      const baseThreshold = BASE_OPEN_THRESHOLDS[position] || 0.50;
      let openThreshold = (baseThreshold / posMult) * 0.85;
      if (hasAnte) openThreshold *= 0.88;
      if (oppType === 'station') openThreshold *= 1.12;
      if (oppType === 'nit') openThreshold *= 0.80;
      if (oppType === 'maniac') openThreshold *= 1.05;
      openThreshold = Math.min(0.88, Math.max(0.12, openThreshold));

      if (position === 'SB') {
        const limpFreq = 0.03;
        const roll = Math.random();
        diagnostics.threshold = openThreshold; diagnostics.limpFreq = limpFreq;
        diagnostics.reason = `SB RFI LAG thr=${openThreshold.toFixed(2)}`;
        if (strength > openThreshold) {
          return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (hasAnte ? 2.2 : 2.6))), diagnostics };
        }
        if (strength > 0.38 && roll < limpFreq) return { action: 'call', amount: bb, diagnostics };
        return { action: 'fold', amount: 0, diagnostics };
      }

      diagnostics.threshold = openThreshold;
      diagnostics.reason = `RFI ${position} LAG thr=${openThreshold.toFixed(2)}`;
      if (strength > openThreshold) {
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * (hasAnte ? 2.0 : 2.4))), diagnostics };
      }
      return { action: 'fold', amount: 0, diagnostics };
    }
  };

  // ========================================================================
  // BLOCK 8 — DISPATCHER & PUBLIC FACADE
  // ========================================================================

  const BOT_IMPLEMENTATIONS = {
    PRO_GTO: PRO_GTO,
    PRO_SOLVER: PRO_SOLVER,
    PRO_ICM: PRO_ICM,
    PRO_LAG: PRO_LAG
  };

  function decidePreflop(ctx) {
    if (!ctx || !ctx.botType) return { action: 'fold', amount: 0, diagnostics: { error: 'missing_context' } };
    const impl = BOT_IMPLEMENTATIONS[ctx.botType];
    if (!impl) return { action: 'fold', amount: 0, diagnostics: { error: 'unknown_bot_type', botType: ctx.botType } };
    return impl.decidePreflop(ctx);
  }

  function decidePostflop(ctx) {
    const profile = PROFILES[ctx.botType];
    if (!profile) return { action: 'fold', amount: 0, diagnostics: { error: 'no_profile' } };
    const card1 = ctx.card1, card2 = ctx.card2;
    const board = ctx.board || [];
    const toCall = ctx.toCall || 0;
    const pot = ctx.pot || 0;
    if (toCall === 0) {
      const strength = preflopStrength(card1, card2);
      if (strength > 0.60) {
        const betSize = Math.round(pot * 0.5);
        return { action: 'bet', amount: Math.min(ctx.myStack, Math.max(ctx.bb, betSize)), diagnostics: { source: 'pro', strength } };
      }
      return { action: 'check', amount: 0, diagnostics: { source: 'pro', strength } };
    }
    const potOdds = toCall / (pot + toCall);
    if (preflopStrength(card1, card2) > potOdds * 1.5) return { action: 'call', amount: toCall, diagnostics: { source: 'pro' } };
    return { action: 'fold', amount: 0, diagnostics: { source: 'pro' } };
  }

  function decide(ctx) {
    if (!ctx || !ctx.botType) return { action: 'fold', amount: 0, diagnostics: { error: 'invalid_ctx' } };
    return ctx.street === 'preflop' ? decidePreflop(ctx) : decidePostflop(ctx);
  }

  function assignBotTypes() {
    return ['PRO_LAG', 'PRO_LAG', 'PRO_LAG', 'PRO_ICM', 'PRO_ICM', 'PRO_SOLVER', 'PRO_GTO'];
  }

  function assignBotsToSeats(botTypes) {
    const arr = botTypes.slice();
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Engine.cryptoRandomInt(i + 1);
      const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    return arr;
  }

  function updateModel() {}
  function updateMeta() {}
  function getRead() { return null; }
  function reset() { opponentStats.clear(); }
  function dumpStats() { return Array.from(opponentStats.entries()); }

  function getProfile(botType) { return PROFILES[botType] || null; }
  function getTypeMeta(botType) { return TYPES[botType] || null; }

  const ProTexture = {
    analyze: function(board) {
      const result = { wet: 0, paired: false, flushPossible: false, straightPossible: false, highCard: 0, flushSuit: null, suitedCount: 0 };
      if (!board || board.length === 0) return result;
      const values = board.map(c => R[c.rank]);
      const suits = board.map(c => c.suit);
      const sorted = values.slice().sort((a, b) => b - a);
      result.highCard = sorted[0] || 0;
      const freq = {};
      for (const v of values) freq[v] = (freq[v] || 0) + 1;
      result.paired = Object.values(freq).some(v => v >= 2);
      const suitFreq = {};
      for (const s of suits) suitFreq[s] = (suitFreq[s] || 0) + 1;
      let suitedCount = 0; let flushSuit = null;
      for (const [s, c] of Object.entries(suitFreq)) {
        if (c > suitedCount) { suitedCount = c; flushSuit = s; }
      }
      result.suitedCount = suitedCount;
      result.flushSuit = flushSuit;
      result.flushPossible = suitedCount >= 3;
      const uniqueVals = [...new Set(values)].sort((a, b) => a - b);
      for (let i = 0; i < uniqueVals.length - 2; i++) {
        if (uniqueVals[i + 2] - uniqueVals[i] <= 4) { result.straightPossible = true; break; }
      }
      let wet = 0;
      if (result.flushPossible) wet += 0.35;
      if (result.straightPossible) wet += 0.35;
      if (result.paired) wet += 0.15;
      if (suitedCount === 4) wet += 0.15;
      result.wet = Math.min(1, wet);
      return result;
    }
  };

  // ========================================================================
  // BLOCK 9 — EXPORT
  // ========================================================================

  const Bots = {
    CLASS_PRO,
    CLASS_VIP: null,
    TYPES, PROFILES, Texture: ProTexture,
    decidePreflop, decidePostflop, decide,
    assignBotTypes, assignBotsToSeats,
    preflopStrength,
    handInChart: function() { return false; },
    updateModel, updateMeta, getRead, reset, dumpStats,
    getProfile, getTypeMeta,
    updateOpponentStats, getOpponentType, getShrunk,
    detectContext,
    _internal: {
      calculateBubbleFactor, getStackDepth, getPositionMultiplier,
      getBlockerScore, equityRealization, shrink, applyStatModifier,
      NASH_PUSH_TABLE, BASE_OPEN_THRESHOLDS, POSITION_MULTIPLIERS,
      PRIORS, SHRINK_K, STAT_CONFIDENCE, opponentStats
    }
  };

  Bots.__level = 'top';
  Bots.__version = '2.2-preflop-context';
  Bots.__source = 'bots.js';

  window.Bots = Bots;
  window.__botsJsLoaded = true;

})();
