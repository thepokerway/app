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
    PRO_GTO:    { aggression: 0.68, looseness: 0.42, bluff: 0.22, callThreshold: 0.55, icmSensitivity: 0.70, positionAdaptivity: 1.00, limpHabit: 0.12 },
    PRO_SOLVER: { aggression: 0.80, looseness: 0.55, bluff: 0.36, callThreshold: 0.45, icmSensitivity: 0.55, positionAdaptivity: 1.15, limpHabit: 0.08 },
    PRO_ICM:    { aggression: 0.72, looseness: 0.45, bluff: 0.28, callThreshold: 0.52, icmSensitivity: 1.30, positionAdaptivity: 1.05, limpHabit: 0.10 },
    PRO_LAG:    { aggression: 0.94, looseness: 0.72, bluff: 0.50, callThreshold: 0.32, icmSensitivity: 0.35, positionAdaptivity: 1.25, limpHabit: 0.05 }
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

  // Realistic PRO open thresholds (normalized hand strength 0..1).
  // These correspond roughly to: UTG 15-17%, HJ 22-25%, CO 28-33%, BTN 42-52%
  const BASE_OPEN_THRESHOLDS = {
    UTG: 0.60, UTG1: 0.58, LJ: 0.53, HJ: 0.48,
    CO: 0.42, BTN: 0.30, SB: 0.36, BB: 0.55
  };

  // Nash push frequencies (approx chip-EV) per position and stack depth
  const NASH_PUSH_TABLE = {
    UTG:  { 5: 0.24, 8: 0.20, 10: 0.175, 12: 0.155, 15: 0.135, 20: 0.115, 25: 0.100 },
    UTG1: { 5: 0.27, 8: 0.23, 10: 0.195, 12: 0.170, 15: 0.150, 20: 0.128, 25: 0.110 },
    LJ:   { 5: 0.32, 8: 0.28, 10: 0.240, 12: 0.205, 15: 0.175, 20: 0.145, 25: 0.125 },
    HJ:   { 5: 0.38, 8: 0.33, 10: 0.285, 12: 0.240, 15: 0.205, 20: 0.170, 25: 0.145 },
    CO:   { 5: 0.46, 8: 0.40, 10: 0.345, 12: 0.290, 15: 0.245, 20: 0.200, 25: 0.170 },
    BTN:  { 5: 0.57, 8: 0.50, 10: 0.435, 12: 0.365, 15: 0.310, 20: 0.255, 25: 0.210 },
    SB:   { 5: 0.70, 8: 0.62, 10: 0.535, 12: 0.450, 15: 0.390, 20: 0.320, 25: 0.260 },
    BB:   { 5: 0.52, 8: 0.44, 10: 0.375, 12: 0.320, 15: 0.270, 20: 0.220, 25: 0.180 }
  };

  // Prior probabilities for Bayesian shrinkage
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

  // Confidence thresholds for exploiting statistics
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
    // Pairs: 0.50..1.00 (22..AA)
    if (v1 === v2) return 0.50 + (hi - 2) / 12 * 0.50;
    // Non-pairs: base from high card + low card contributions
    let s = (hi - 2) / 12 * 0.50 + (lo - 2) / 12 * 0.20;
    // Suited bonus
    if (suited) s += 0.08;
    // Ace blocker / wheel potential
    if (hi === 14) s += 0.03;
    // Connectedness
    const gap = Math.abs(hi - lo);
    if (gap === 1) s += 0.05;
    else if (gap === 2) s += 0.03;
    else if (gap === 3) s += 0.01;
    // Broadway hands still valuable but not as much as suited
    if (!suited && hi >= 10 && lo >= 10) s += 0.02;
    // Offsuit broadway penalty (play worse than strength suggests)
    if (!suited && hi >= 11 && lo >= 10) s -= 0.03;
    // Very weak offsuit hands penalty
    if (!suited && hi <= 10 && gap >= 4) s -= 0.03;
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
    const suited = card1.suit === card2.suit;
    let score = 0;
    // A5s-A2s: block AA/AK, retain equity
    if (v1 === 14 && v2 >= 2 && v2 <= 5 && suited) score += 0.15;
    // A9s-A6s: medium
    if (v1 === 14 && v2 >= 6 && v2 <= 9 && suited) score += 0.05;
    // KQs-KTs: block KK/QQ/AK/AQ
    if (v1 === 13 && v2 >= 10 && v2 <= 12 && suited) score += 0.06;
    // QJs, JTs: weak, block folds
    if ((v1 === 12 && v2 === 11 && suited) || (v1 === 11 && v2 === 10 && suited)) score -= 0.10;
    // Small pairs, connectors: no meaningful blocker
    if (v1 === v2 && v1 <= 8) score -= 0.20;
    // A9o on bubble: strong blocker, better than 98s
    if (v1 === 14 && v2 === 9 && !suited) score += 0.10;
    // Offsuit A-x: medium blocker
    if (v1 === 14 && v2 >= 2 && v2 <= 8 && !suited) score += 0.05;
    return score;
  }

  function getBlockerReason(card1, card2) {
    const v1 = R[card1.rank], v2 = R[card2.rank];
    const suited = card1.suit === card2.suit;
    if (v1 === 14 && v2 >= 2 && v2 <= 5 && suited) return 'A5s-A2s: blocks AA/AK';
    if (v1 === 14 && v2 === 9 && !suited) return 'A9o: blocks AA/AK/AQ';
    if (v1 === 14 && v2 >= 2 && v2 <= 8) return 'A-x: partial AA/AK block';
    if (v1 === 13 && v2 >= 10 && v2 <= 12 && suited) return 'KQs-KTs: blocks KK/QQ/AK';
    if ((v1 === 12 && v2 === 11) || (v1 === 11 && v2 === 10)) return 'QJs/JTs: weak blocker';
    if (v1 === v2 && v1 <= 8) return 'small pair: no blocker';
    return '';
  }

  function equityRealization(card1, card2, position, numActive) {
    let mult = 1.0;
    const v1 = R[card1.rank], v2 = R[card2.rank];
    const suited = card1.suit === card2.suit;
    const gap = Math.abs(v1 - v2);
    const isConnector = gap === 1;
    const isSmallPair = v1 === v2 && v1 <= 8;
    const isBroadwayOffsuit = !suited && v1 >= 10 && v2 >= 10;

    // Position factor
    if (position === 'BTN' || position === 'CO') mult *= 1.08;
    else if (position === 'SB' || position === 'BB') mult *= 0.88;
    else mult *= 0.97;

    // Hand structure
    if (suited) mult += 0.06;
    if (isConnector) mult += 0.03;
    if (isBroadwayOffsuit) mult -= 0.08;
    if (isSmallPair && (position === 'SB' || position === 'BB')) mult -= 0.05;

    // Multiway penalty
    if (numActive > 2) mult -= 0.06;
    if (numActive > 4) mult -= 0.04;

    return Math.max(0.65, Math.min(1.18, mult));
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
    if (stats.handsPlayed < 15) return 'unknown';
    const vpip = shrink(stats.coldCall + stats.limp, stats.coldCallOpportunities || 1, 'coldCall').value;
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

  function calculateBubbleFactor(myStackBB, opponentStackBB, playersLeft, itmCount, payoutStructure) {
    if (playersLeft <= itmCount) return 1.0;
    const onBubble = (playersLeft - itmCount) <= 2;
    const finalTable = playersLeft <= 3;
    let base = 1.0;
    if (finalTable) {
      base = 1.85 + (payoutStructure === 'steep' ? 0.25 : 0.05);
    } else if (onBubble) {
      base = 1.65 + (payoutStructure === 'steep' ? 0.20 : 0.05);
    } else if (playersLeft <= itmCount + 4) {
      base = 1.30;
    } else {
      base = 1.10;
    }
    // Short stack BF is lower (can push wider, fold equity high)
    if (myStackBB < 15) base *= 0.85;
    // Big stack BF is lower (can pressure, opponent can't call wide)
    else if (myStackBB > 40) base *= 0.90;
    // If opponent is bigger stack and can call without risk, tighten further
    if (opponentStackBB > myStackBB * 1.5) base *= 1.10;
    return Math.max(1.0, base);
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

  // Apply stat modifier to a base threshold. Returns adjusted + diagnostics.
  function applyStatModifier(baseValue, stat, direction, multiplier) {
    const m = (multiplier !== undefined) ? multiplier : 0.15;
    const prior = stat.value;
    const conf = stat.confidence;
    let effect = 0;
    let level = 'ignored';

    if (conf >= STAT_CONFIDENCE.IGNORE) {
      const delta = (prior - 0.5);
      let rawEffect;
      if (direction === 'fold_to_aggr') rawEffect = -delta;
      else if (direction === 'call_station') rawEffect = delta;
      else if (direction === 'respect_aggr') rawEffect = delta;
      else rawEffect = delta;

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
  // BLOCK 4 — CONTEXT DETECTION
  // ========================================================================

  // Detect the actual preflop/postflop situation, ignoring "toCall > 0" traps.
  function detectContext(ctx) {
    const street = ctx.street;
    const position = ctx.position;
    const toCall = ctx.toCall || 0;
    const bb = ctx.bb || 1;
    const currentBet = ctx.currentBet || 0;
    const pot = ctx.pot || 0;
    const numActive = ctx.numActive || 8;

    if (street !== 'preflop') {
      if (toCall > 0) {
        if (ctx.facingAllin) return 'facing_allin_postflop';
        return 'facing_bet_postflop';
      }
      return 'unopened_postflop';
    }

    const isBB = position === 'BB';
    const isSB = position === 'SB';

    // Count limpers
    let numLimpers = 0;
    if (ctx.numLimpers !== undefined) {
      numLimpers = ctx.numLimpers;
    } else {
      // Cannot reliably count without engine support; use pot heuristic
      if (currentBet === bb && pot > bb * 2.5) {
        numLimpers = Math.max(1, Math.round((pot - bb) / bb) - 1);
      }
    }

    const isRealRaise = currentBet > bb;

    // BB check option: BB has already paid, no raise exists, no call required
    if (isBB && !isRealRaise && toCall === 0) {
      if (numLimpers > 0) return 'bb_vs_limp';
      return 'bb_check_option';
    }

    // SB limp option (SB called to limp, BB gets option)
    if (isBB && !isRealRaise && toCall === 0) {
      return 'bb_check_option';
    }

    // Facing all-in
    if (ctx.facingAllin) return 'facing_allin_preflop';

    // No raise yet
    if (!isRealRaise) {
      if (numLimpers > 0 && toCall > 0) return 'facing_limp';
      if (numLimpers > 0 && toCall === 0) return 'bb_vs_limp';
      return 'unopened';
    }

    // Real raise exists
    const raiseLevel = currentBet / bb;
    if (raiseLevel >= 10.0) return 'facing_4bet';
    if (raiseLevel >= 4.5) return 'facing_3bet';
    return 'facing_raise';
  }

  // ========================================================================
  // BLOCK 5 — SHARED HELPERS
  // ========================================================================

  function getOpenSizeBB(ctx) {
    const position = ctx.position;
    const stackBB = (ctx.myStack || 0) / (ctx.bb || 1);
    const hasAnte = (ctx.ante > 0) || (ctx.blinds && ctx.blinds.ante > 0);
    let size = 2.5;
    if (stackBB < 40) size = 2.3;
    if (stackBB < 25) size = 2.1;
    if (stackBB < 18) size = 2.0;
    if (position === 'SB') size += 0.3;
    if (position === 'BTN') size = Math.max(2.0, size - 0.2);
    if (hasAnte) size = Math.max(2.0, size - 0.4);
    return size;
  }

  function getThreeBetSizeBB(ctx, openerSizeBB, isIP) {
    const hasAnte = (ctx.ante > 0) || (ctx.blinds && ctx.blinds.ante > 0);
    let mult = isIP ? 3.4 : 4.4;
    const stackBB = (ctx.myStack || 0) / (ctx.bb || 1);
    if (stackBB < 50) mult = isIP ? 3.0 : 4.0;
    if (stackBB < 30) mult = isIP ? 2.8 : 3.6;
    let size = openerSizeBB * mult;
    if (hasAnte) size += 0.3;
    return size;
  }

  function getSqueezeSizeBB(ctx, openerSizeBB, numCallers, isOOP) {
    let size = openerSizeBB * 3.7 + numCallers * 1.0;
    if (isOOP) size += 1.0;
    return size;
  }

  function getIsoSizeBB(ctx, numLimpers) {
    const hasAnte = (ctx.ante > 0) || (ctx.blinds && ctx.blinds.ante > 0);
    let size = 3.0 + numLimpers * 1.0;
    if (hasAnte) size += 0.3;
    return size;
  }

  function isInPosition(position, openerPosition) {
    const order = { 'UTG': 0, 'UTG1': 1, 'LJ': 2, 'HJ': 3, 'CO': 4, 'BTN': 5, 'SB': 6, 'BB': 7 };
    const myIdx = order[position] !== undefined ? order[position] : 0;
    const opIdx = order[openerPosition] !== undefined ? order[openerPosition] : 5;
    // BTN is always IP vs blinds; blinds are OOP vs everyone
    if (position === 'BTN' || position === 'CO') return true;
    if (position === 'SB' || position === 'BB') return false;
    return myIdx > opIdx;
  }

  // ========================================================================
  // BLOCK 6 — SHARED PREFLOP DECISION FRAMEWORK
  // ========================================================================
  // Each bot type uses the same framework but with different profile params.
  // This ensures consistent context detection and shared improvement.

  function decidePreflopCore(botType, ctx, profile) {
    const card1 = ctx.card1, card2 = ctx.card2;
    if (!card1 || !card2) {
      return { action: 'fold', amount: 0, diagnostics: { error: 'missing_cards' } };
    }
    const position = ctx.position;
    const myStack = ctx.myStack || 0;
    const bb = ctx.bb || 1;
    const toCall = ctx.toCall || 0;
    const pot = ctx.pot || 0;
    const stackBB = myStack / bb;
    const stackDepth = getStackDepth(stackBB);
    const rawStrength = preflopStrength(card1, card2);
    const posMult = getPositionMultiplier(position);
    const blockerScore = getBlockerScore(card1, card2);
    const blockerReason = getBlockerReason(card1, card2);
    const hasAnte = (ctx.ante > 0) || (ctx.blinds && ctx.blinds.ante > 0);

    const numActive = ctx.numActive || 8;
    const er = equityRealization(card1, card2, position, numActive);
    const strength = Math.min(1, rawStrength * er);

    const playersLeft = ctx.playersLeft || 8;
    const itmCount = ctx.itmCount || 3;
    const bubbleFactor = calculateBubbleFactor(stackBB, ctx.opponentStackBB || stackBB, playersLeft, itmCount, 'steep');
    const icmPressure = getICMPressure(playersLeft, itmCount);
    const payoutStructure = getPayoutStructure(playersLeft, itmCount);

    const contextType = detectContext(ctx);
    const numLimpers = ctx.numLimpers !== undefined ? ctx.numLimpers : 0;

    const oppId = ctx.opponentId || null;
    const oppType = oppId ? getOpponentType(oppId) : 'unknown';
    const foldToResteal = oppId ? getShrunk(oppId, 'foldToResteal') : { value: PRIORS.foldToResteal, confidence: 0, sampleSize: 0 };
    const foldTo4Bet = oppId ? getShrunk(oppId, 'foldTo4Bet') : { value: PRIORS.foldTo4Bet, confidence: 0, sampleSize: 0 };
    const foldToSteal = oppId ? getShrunk(oppId, 'foldBbToStealBtn') : { value: PRIORS.foldBbToStealBtn, confidence: 0, sampleSize: 0 };

    const diagnostics = {
      source: 'pro',
      botType: botType,
      decisionBranch: contextType,
      contextType: contextType,
      strength: strength,
      strengthRaw: rawStrength,
      equityRealization: er,
      equityRaw: rawStrength,
      equityRealized: strength,
      equityVsRange: null,
      stackDepth: stackDepth,
      position: position,
      positionModifier: posMult,
      effectiveStackBB: stackBB,
      bubbleFactor: bubbleFactor,
      icmPressure: icmPressure,
      payoutStructure: payoutStructure,
      blockerScore: blockerScore,
      blockerReason: blockerReason,
      multiwayCount: numActive,
      opponentType: oppType,
      exploitActive: false,
      statUsed: null,
      statKey: null,
      statConfidence: 0,
      statEffect: 0,
      statSampleSize: 0,
      hasAnte: hasAnte,
      frequency: 1.0,
      randomRoll: null,
      threshold: null,
      sizingReason: '',
      limpingReason: '',
      metaLevel: 1,
      adaptationDetected: false,
      numLimpers: numLimpers,
      raiserPosition: ctx.raiserPosition || null,
      isRealRaise: (ctx.currentBet || 0) > bb,
      potOdds: null,
      foldEquity: null,
      spr: null,
      evFold: 0,
      evCall: 0,
      evRaise: 0,
      reason: ''
    };

    // ===== FACING ALL-IN =====
    if (contextType === 'facing_allin_preflop') {
      let callThreshold = 0.55 + (bubbleFactor - 1) * profile.icmSensitivity * 0.15;
      callThreshold = Math.min(0.95, callThreshold);
      if (icmPressure === 'bubble' && stackBB > 15 && stackBB < 35) callThreshold += 0.10;
      if (icmPressure === 'final_table' && stackBB > 15 && stackBB < 35) callThreshold += 0.08;
      // Blocker helps against shoves on bubble (A5s better than 98s)
      if (icmPressure !== 'none') callThreshold -= blockerScore * 0.5;
      diagnostics.threshold = callThreshold;
      diagnostics.reason = `facing allin: need ${(callThreshold*100).toFixed(0)}% (BF=${bubbleFactor.toFixed(2)}, ${icmPressure})`;
      if (strength > callThreshold) {
        diagnostics.reason += ' | call';
        return { action: 'call', amount: Math.min(toCall, myStack), diagnostics };
      }
      diagnostics.reason += ' | fold';
      return { action: 'fold', amount: 0, diagnostics };
    }

    // ===== SHORT STACK PUSH/FOLD (unopened only) =====
    if (contextType === 'unopened' && stackBB <= 20 && toCall <= bb) {
      const table = NASH_PUSH_TABLE[position] || NASH_PUSH_TABLE.BTN;
      let pushFreq = 0.10;
      for (const s of [5, 8, 10, 12, 15, 20, 25]) {
        if (stackBB <= s) { pushFreq = table[s] || pushFreq; break; }
      }
      if (hasAnte) pushFreq *= 1.08;
      if (icmPressure === 'bubble' && stackBB < 15) pushFreq *= 1.15;
      if (icmPressure === 'final_table' && stackBB < 15) pushFreq *= 1.10;
      if (icmPressure === 'bubble' && stackBB >= 15 && stackBB <= 30) pushFreq *= 0.78;
      if (icmPressure === 'final_table' && stackBB >= 15 && stackBB <= 30) pushFreq *= 0.82;
      if (profile.aggression > 0.85) pushFreq *= 1.10;
      if (profile.aggression < 0.5) pushFreq *= 0.90;
      pushFreq = Math.min(1, pushFreq);

      const roll = Math.random();
      diagnostics.frequency = pushFreq;
      diagnostics.randomRoll = roll;
      diagnostics.threshold = 'nash_push';
      diagnostics.reason = `Nash push ${position} ${stackBB.toFixed(0)}bb freq=${pushFreq.toFixed(2)} ${icmPressure}`;
      if (roll < pushFreq && strength > 0.28) {
        diagnostics.reason += ' | ALL-IN';
        return { action: 'allin', amount: myStack, diagnostics };
      }
      diagnostics.reason += ' | fold';
      return { action: 'fold', amount: 0, diagnostics };
    }

    // ===== BB CHECK OPTION =====
    if (contextType === 'bb_check_option') {
      // BB has no call to make. Only check (no fold is allowed).
      // Consider iso-raise with strong hands over limpers - handled in bb_vs_limp.
      diagnostics.threshold = 'check';
      diagnostics.reason = 'BB check option (no raise, no call needed)';
      return { action: 'check', amount: 0, diagnostics };
    }

    // ===== BB VS LIMP =====
    if (contextType === 'bb_vs_limp') {
      const limpers = Math.max(1, numLimpers);
      // Iso-raise threshold scales with position and limpers
      let isoThreshold = 0.55;
      if (hasAnte) isoThreshold -= 0.03;
      if (limpers >= 2) isoThreshold -= 0.03;
      if (profile.aggression > 0.85) isoThreshold -= 0.08;
      if (profile.aggression < 0.5) isoThreshold += 0.05;
      isoThreshold = Math.max(0.35, Math.min(0.80, isoThreshold));

      diagnostics.threshold = isoThreshold;
      diagnostics.reason = `BB vs ${limpers} limp(s) thr=${isoThreshold.toFixed(2)}`;

      if (strength > isoThreshold) {
        const sizeBB = getIsoSizeBB(ctx, limpers);
        diagnostics.sizingReason = `iso-raise ${limpers} limper(s) = ${sizeBB.toFixed(1)}bb`;
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * sizeBB)), diagnostics };
      }
      diagnostics.reason += ' | check';
      return { action: 'check', amount: 0, diagnostics };
    }

    // ===== FACING LIMP (not BB) =====
    if (contextType === 'facing_limp') {
      const limpers = Math.max(1, numLimpers);
      // Iso-raise threshold: positive aggression profiles iso wider
      let isoThreshold = 0.42;
      if (position === 'SB') isoThreshold += 0.06;
      if (position === 'BTN' || position === 'CO') isoThreshold -= 0.05;
      if (hasAnte) isoThreshold -= 0.02;
      if (limpers >= 2) isoThreshold -= 0.03;
      if (profile.aggression > 0.85) isoThreshold -= 0.06;
      if (profile.aggression < 0.5) isoThreshold += 0.04;
      isoThreshold = Math.max(0.25, Math.min(0.72, isoThreshold));

      diagnostics.threshold = isoThreshold;
      diagnostics.reason = `iso-raise vs ${limpers} limper(s) thr=${isoThreshold.toFixed(2)}`;

      if (strength > isoThreshold) {
        const sizeBB = getIsoSizeBB(ctx, limpers);
        diagnostics.sizingReason = `iso-raise ${limpers} limper(s) = ${sizeBB.toFixed(1)}bb`;
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * sizeBB)), diagnostics };
      }
      // Small blind can complete if strength is decent
      if (position === 'SB' && strength > 0.30 && toCall <= bb) {
        diagnostics.reason += ' | SB complete';
        diagnostics.limpingReason = 'pot_odds_complete';
        return { action: 'call', amount: toCall, diagnostics };
      }
      diagnostics.reason += ' | fold';
      return { action: 'fold', amount: 0, diagnostics };
    }

    // ===== FACING RAISE (2-bet) =====
    if (contextType === 'facing_raise') {
      const openPos = ctx.raiserPosition || 'BTN';
      const openerSizeBB = (ctx.currentBet || bb) / bb;
      const isIP = isInPosition(position, openPos);

      // Three-bet decision
      let threeBetThreshold = 0.68;
      let threeBetFreqIfBlocker = 0.25;
      if (oppType === 'nit') { threeBetThreshold = 0.75; threeBetFreqIfBlocker = 0.15; }
      if (oppType === 'maniac') { threeBetThreshold = 0.60; threeBetFreqIfBlocker = 0.35; }
      if (oppType === 'lag') { threeBetThreshold = 0.65; threeBetFreqIfBlocker = 0.30; }

      // Stat-based: foldToResteal confidence
      if (foldToResteal.confidence >= STAT_CONFIDENCE.IGNORE) {
        const statAdj = applyStatModifier(threeBetThreshold, foldToResteal, 'fold_to_aggr', 0.15);
        threeBetThreshold = statAdj.value;
        diagnostics.statUsed = 'foldToResteal';
        diagnostics.statConfidence = foldToResteal.confidence;
        diagnostics.statEffect = statAdj.effect;
        diagnostics.statSampleSize = foldToResteal.sampleSize;
        if (statAdj.level !== 'ignored') diagnostics.exploitActive = true;
      }

      const roll = Math.random();
      if (strength > threeBetThreshold) {
        const sizeBB = getThreeBetSizeBB(ctx, openerSizeBB, isIP);
        diagnostics.threshold = threeBetThreshold;
        diagnostics.frequency = 1.0;
        diagnostics.randomRoll = roll;
        diagnostics.sizingReason = `3bet ${isIP ? 'IP' : 'OOP'} x${(sizeBB/openerSizeBB).toFixed(1)}`;
        diagnostics.reason = `3bet vs ${openPos} strength=${strength.toFixed(2)}>thr=${threeBetThreshold.toFixed(2)}`;
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * sizeBB)), diagnostics };
      }
      if (blockerScore > 0.08 && roll < threeBetFreqIfBlocker) {
        const sizeBB = getThreeBetSizeBB(ctx, openerSizeBB, isIP);
        diagnostics.threshold = 'blocker_bluff';
        diagnostics.frequency = threeBetFreqIfBlocker;
        diagnostics.randomRoll = roll;
        diagnostics.sizingReason = `blocker bluff 3bet`;
        diagnostics.reason = `3bet blocker bluff (score=${blockerScore.toFixed(2)}) vs ${openPos}`;
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * sizeBB)), diagnostics };
      }

      // Call defense thresholds
      let callThreshold = 0.40;
      if (position === 'BB') {
        if (openPos === 'UTG' || openPos === 'UTG1') callThreshold = 0.48;
        else if (openPos === 'LJ' || openPos === 'HJ') callThreshold = 0.40;
        else if (openPos === 'CO') callThreshold = 0.32;
        else callThreshold = 0.28;
        if (hasAnte) callThreshold -= 0.05;
      } else if (position === 'SB') {
        // SB strictly 3bet or fold (except rare balance)
        callThreshold = 0.99;
      } else {
        const potOdds = toCall / (pot + toCall);
        callThreshold = Math.max(0.40, potOdds * 1.4);
        // Late position: can call wider vs earlier opener (smaller relative weakness)
        if (isIP) callThreshold -= 0.03;
      }

      // ICM adjustments
      if (icmPressure === 'bubble' && stackBB > 15 && stackBB < 35) callThreshold += 0.08;
      if (icmPressure === 'final_table' && stackBB > 15 && stackBB < 35) callThreshold += 0.06;

      // Multiway: tighten
      if (numActive < 4) callThreshold += 0.03;

      diagnostics.threshold = callThreshold;
      diagnostics.potOdds = toCall / (pot + toCall);

      if (strength > callThreshold) {
        diagnostics.reason = `call vs ${openPos} thr=${callThreshold.toFixed(2)}${isIP ? ' (IP)' : ' (OOP)'}`;
        return { action: 'call', amount: Math.min(toCall, myStack), diagnostics };
      }
      diagnostics.reason = `fold vs ${openPos} thr=${callThreshold.toFixed(2)}`;
      return { action: 'fold', amount: 0, diagnostics };
    }

    // ===== FACING 3-BET =====
    if (contextType === 'facing_3bet') {
      let fourBetThreshold = 0.82;
      let fourBetFreq = 0.65;
      if (oppType === 'maniac') { fourBetThreshold = 0.72; fourBetFreq = 0.80; }
      if (oppType === 'nit') { fourBetThreshold = 0.88; fourBetFreq = 0.50; }

      if (foldTo4Bet.confidence >= STAT_CONFIDENCE.MICRO) {
        if (foldTo4Bet.value > 0.60) {
          fourBetThreshold = Math.max(0.65, fourBetThreshold - 0.08);
          fourBetFreq = Math.max(fourBetFreq, 0.80);
          diagnostics.statUsed = 'foldTo4Bet';
          diagnostics.statConfidence = foldTo4Bet.confidence;
          diagnostics.statEffect = -0.08;
          diagnostics.exploitActive = true;
        } else if (foldTo4Bet.value < 0.30) {
          fourBetThreshold = Math.min(0.92, fourBetThreshold + 0.05);
          fourBetFreq = Math.min(fourBetFreq, 0.40);
          diagnostics.statUsed = 'foldTo4Bet';
          diagnostics.statConfidence = foldTo4Bet.confidence;
          diagnostics.statEffect = 0.05;
          diagnostics.exploitActive = true;
        }
      }

      const roll = Math.random();
      diagnostics.threshold = fourBetThreshold;
      diagnostics.frequency = fourBetFreq;
      diagnostics.randomRoll = roll;

      if (strength > fourBetThreshold && roll < fourBetFreq && stackBB > 15) {
        const sizeBB = (ctx.currentBet || bb * 8) * 2.2 / bb;
        diagnostics.sizingReason = `4bet ×2.2 of 3bet`;
        diagnostics.reason = `4bet vs 3bet thr=${fourBetThreshold.toFixed(2)}`;
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * sizeBB)), diagnostics };
      }
      // IP flat with strong hands
      const isIP = isInPosition(position, ctx.raiserPosition || 'BTN');
      if (isIP && strength > 0.55 && stackBB > 25) {
        diagnostics.reason = `flat IP vs 3bet`;
        return { action: 'call', amount: Math.min(toCall, myStack), diagnostics };
      }
      diagnostics.reason = `fold vs 3bet`;
      return { action: 'fold', amount: 0, diagnostics };
    }

    // ===== FACING 4-BET =====
    if (contextType === 'facing_4bet') {
      let fiveBetThreshold = 0.88;
      if (foldTo4Bet.confidence >= STAT_CONFIDENCE.MICRO && foldTo4Bet.value > 0.60) {
        fiveBetThreshold = 0.80;
        diagnostics.statUsed = 'foldTo4Bet';
        diagnostics.statConfidence = foldTo4Bet.confidence;
        diagnostics.statEffect = -0.08;
        diagnostics.exploitActive = true;
      }
      diagnostics.threshold = fiveBetThreshold;
      if (strength > fiveBetThreshold && stackBB > 20) {
        diagnostics.reason = `5bet allin vs 4bet thr=${fiveBetThreshold.toFixed(2)}`;
        return { action: 'allin', amount: myStack, diagnostics };
      }
      if (strength > 0.75 && stackBB < 20) {
        diagnostics.reason = `allin vs 4bet short`;
        return { action: 'allin', amount: myStack, diagnostics };
      }
      diagnostics.reason = `fold vs 4bet`;
      return { action: 'fold', amount: 0, diagnostics };
    }

    // ===== UNOPENED (RFI) =====
    const baseThreshold = BASE_OPEN_THRESHOLDS[position] || 0.50;
    // We use base threshold directly now - the base is already position-aware
    let openThreshold = baseThreshold;
    // Aggression profile
    openThreshold += (0.68 - profile.aggression) * 0.12;
    // Ante: open slightly wider (uneven by position)
    if (hasAnte) {
      if (position === 'UTG' || position === 'UTG1') openThreshold -= 0.02;
      else if (position === 'BTN' || position === 'SB') openThreshold -= 0.07;
      else openThreshold -= 0.04;
    }
    // ICM
    if (icmPressure === 'bubble') {
      if (stackBB > 30) openThreshold -= 0.08;
      else if (stackBB > 15) openThreshold += 0.10;
    }
    if (icmPressure === 'final_table') {
      if (stackBB > 30) openThreshold -= 0.10;
      else if (stackBB > 15) openThreshold += 0.12;
    }
    // Exploit vs opener's steal stats (if opener exists in future streets) - not directly applicable here.
    openThreshold = Math.max(0.10, Math.min(0.95, openThreshold));

    // SB: mixed strategy - raise / limp / fold (GTO limp present)
    if (position === 'SB') {
      // Limp frequency scales with profile, ante, and BB tendencies
      let limpFreq = profile.limpHabit * (hasAnte ? 1.6 : 1.0);
      // If BB is very aggressive 3-bettor, don't limp as much
      if (foldToResteal.confidence >= STAT_CONFIDENCE.IGNORE && foldToResteal.value > 0.55) {
        limpFreq *= 0.6; // BB doesn't fold much, don't open up by limping
      }
      if (icmPressure === 'bubble') limpFreq *= 0.5;
      // SB limp freq caps at 0.30 (don't overdo it)
      limpFreq = Math.min(0.30, limpFreq);

      const roll = Math.random();
      diagnostics.frequency = 1.0;
      diagnostics.threshold = openThreshold;
      diagnostics.randomRoll = roll;

      if (strength > openThreshold) {
        // Some very strong hands limp for balance (rare, only deep stacks with antes)
        const balanceLimpChance = hasAnte && stackBB > 60 ? 0.10 : 0.02;
        if (roll < balanceLimpChance && strength > 0.85) {
          diagnostics.reason = 'SB balance limp (strong)';
          diagnostics.limpingReason = 'balance_limp';
          return { action: 'call', amount: Math.min(bb, myStack), diagnostics };
        }
        const sizeBB = getOpenSizeBB(ctx);
        diagnostics.sizingReason = `SB open ${sizeBB.toFixed(1)}bb`;
        diagnostics.reason = `SB RFI thr=${openThreshold.toFixed(2)}`;
        return { action: 'raise', amount: Math.min(myStack, Math.round(bb * sizeBB)), diagnostics };
      }
      // Speculative limp for medium hands
      if (strength > 0.32 && roll < limpFreq) {
        diagnostics.reason = `SB speculative limp (strength=${strength.toFixed(2)})`;
        diagnostics.limpingReason = 'pot_odds_limp';
        return { action: 'call', amount: Math.min(bb, myStack), diagnostics };
      }
      diagnostics.reason = `SB fold thr=${openThreshold.toFixed(2)}`;
      return { action: 'fold', amount: 0, diagnostics };
    }

    // BTN: mixed limp at mid-short stacks (rare but part of GTO)
    if (position === 'BTN' && stackBB >= 10 && stackBB <= 30) {
      const roll = Math.random();
      if (strength > openThreshold * 0.85 && strength < openThreshold * 1.05 && roll < 0.15) {
        diagnostics.reason = 'BTN mixed limp (balance)';
        diagnostics.limpingReason = 'balance_limp_btn';
        return { action: 'call', amount: Math.min(bb, myStack), diagnostics };
      }
    }

    // Standard RFI
    diagnostics.threshold = openThreshold;
    diagnostics.reason = `RFI ${position} thr=${openThreshold.toFixed(2)}`;
    if (strength > openThreshold) {
      const sizeBB = getOpenSizeBB(ctx);
      diagnostics.sizingReason = `open ${sizeBB.toFixed(1)}bb ${hasAnte ? '(ante)' : ''}`;
      return { action: 'raise', amount: Math.min(myStack, Math.round(bb * sizeBB)), diagnostics };
    }
    return { action: 'fold', amount: 0, diagnostics };
  }

  // ========================================================================
  // BLOCK 7 — PRO_GTO: Baseline, balanced, mixed frequencies
  // ========================================================================

  const PRO_GTO = {
    key: 'PRO_GTO',
    name: 'GTO Baseline',

    decidePreflop: function(ctx) {
      const profile = PROFILES.PRO_GTO;
      const result = decidePreflopCore('PRO_GTO', ctx, profile);
      // GTO: minimal exploit, keep as-is
      return result;
    }
  };

  // ========================================================================
  // BLOCK 8 — PRO_SOLVER: Aggressive, blocker-aware, fast exploit
  // ========================================================================

  const PRO_SOLVER = {
    key: 'PRO_SOLVER',
    name: 'Solver Aggressive',

    decidePreflop: function(ctx) {
      const profile = PROFILES.PRO_SOLVER;
      const result = decidePreflopCore('PRO_SOLVER', ctx, profile);
      // Solver: faster exploit, more bluff 3bet with blockers, slightly wider opens
      if (result.diagnostics && result.diagnostics.decisionBranch === 'facing_raise') {
        // Handled in core (3bet freq uses profile.aggression already)
      }
      return result;
    }
  };

  // ========================================================================
  // BLOCK 9 — PRO_ICM: ICM maximal, bubble specialist
  // ========================================================================

  const PRO_ICM = {
    key: 'PRO_ICM',
    name: 'ICM Specialist',

    decidePreflop: function(ctx) {
      const profile = PROFILES.PRO_ICM;
      const result = decidePreflopCore('PRO_ICM', ctx, profile);
      // ICM bot: extra tightening on bubble/final for open thresholds
      if (result.diagnostics && result.diagnostics.decisionBranch === 'unopened') {
        const stackBB = result.diagnostics.effectiveStackBB;
        const icmPressure = result.diagnostics.icmPressure;
        // Additional tightening beyond core
        if (icmPressure === 'bubble' && stackBB > 15 && stackBB < 35) {
          const extra = 0.10;
          if (result.action === 'raise' && result.diagnostics.threshold &&
              typeof result.diagnostics.threshold === 'number' &&
              result.diagnostics.strength < result.diagnostics.threshold + extra) {
            result.action = 'fold';
            result.amount = 0;
            result.diagnostics.reason += ` | ICM extra-tighten (bubble, ${stackBB.toFixed(0)}bb)`;
          }
        }
      }
      return result;
    }
  };

  // ========================================================================
  // BLOCK 10 — PRO_LAG: Max aggression, exploit-heavy
  // ========================================================================

  const PRO_LAG = {
    key: 'PRO_LAG',
    name: 'LAG Aggressor',

    decidePreflop: function(ctx) {
      const profile = PROFILES.PRO_LAG;
      const result = decidePreflopCore('PRO_LAG', ctx, profile);
      // LAG ignores ICM mostly (profile.icmSensitivity is low)
      return result;
    }
  };

  // ========================================================================
  // BLOCK 11 — POSTFLOP (basic, not Stage 2 — Stage 2 separate)
  // ========================================================================

  function postflopStrength(handScore) {
    if (!handScore) return 0;
    const base = [0.12, 0.30, 0.50, 0.68, 0.78, 0.85, 0.92, 0.97, 0.99, 1.00][handScore.category] || 0;
    const rank = handScore.ranks[0] || 0;
    return Math.min(1, base + (rank / 14) * 0.04);
  }

  function analyzeBoardTexture(board) {
    if (!board || board.length === 0) return { wet: 0, paired: false, flushPossible: false, straightPossible: false, suitedCount: 0 };
    const values = board.map(c => R[c.rank]);
    const suits = board.map(c => c.suit);
    const freq = {};
    for (const v of values) freq[v] = (freq[v] || 0) + 1;
    const paired = Object.values(freq).some(v => v >= 2);
    const suitFreq = {};
    for (const s of suits) suitFreq[s] = (suitFreq[s] || 0) + 1;
    let suitedCount = 0;
    for (const c of Object.values(suitFreq)) if (c > suitedCount) suitedCount = c;
    const flushPossible = suitedCount >= 3;
    const uniqueVals = [...new Set(values)].sort((a, b) => a - b);
    let straightPossible = false;
    for (let i = 0; i < uniqueVals.length - 2; i++) {
      if (uniqueVals[i + 2] - uniqueVals[i] <= 4) { straightPossible = true; break; }
    }
    let wet = 0;
    if (flushPossible) wet += 0.35;
    if (straightPossible) wet += 0.35;
    if (paired) wet += 0.15;
    if (suitedCount === 4) wet += 0.15;
    return { wet: Math.min(1, wet), paired, flushPossible, straightPossible, suitedCount };
  }

  function getMadeHandClass(score) {
    if (!score) return 'nothing';
    const names = ['high_card', 'pair', 'two_pair', 'trips', 'straight', 'flush', 'full_house', 'quads', 'straight_flush', 'royal_flush'];
    return names[score.category] || 'unknown';
  }

  function getDrawClass(card1, card2, board) {
    if (!board || board.length < 3) return 'none';
    const allCards = [card1, card2, ...board];
    const suits = {};
    for (const c of allCards) suits[c.suit] = (suits[c.suit] || 0) + 1;
    let hasFlushDraw = false;
    for (const s of Object.keys(suits)) {
      if (suits[s] === 4) hasFlushDraw = true;
    }
    const values = [...new Set(allCards.map(c => R[c.rank]))].sort((a,b) => a-b);
    let hasStraightDraw = false;
    for (let i = 0; i < values.length - 3; i++) {
      if (values[i + 3] - values[i] <= 4) { hasStraightDraw = true; break; }
    }
    if (hasFlushDraw && hasStraightDraw) return 'combo_draw';
    if (hasFlushDraw) return 'flush_draw';
    if (hasStraightDraw) return 'straight_draw';
    return 'none';
  }

  function decidePostflop(ctx) {
    const card1 = ctx.card1, card2 = ctx.card2;
    if (!card1 || !card2) {
      return { action: 'fold', amount: 0, diagnostics: { error: 'missing_cards' } };
    }
    const board = ctx.board || [];
    if (board.length < 3) {
      return { action: 'check', amount: 0, diagnostics: { error: 'no_flop' } };
    }

    const allCards = [card1, card2, ...board];
    const handScore = Engine.evaluateHand(allCards);
    const strength = postflopStrength(handScore);
    const madeHandClass = getMadeHandClass(handScore);
    const drawClass = getDrawClass(card1, card2, board);
    const texture = analyzeBoardTexture(board);

    const toCall = ctx.toCall || 0;
    const pot = ctx.pot || 0;
    const myStack = ctx.myStack || 0;
    const bb = ctx.bb || 1;
    const street = ctx.street || 'flop';
    const potOdds = toCall > 0 ? toCall / (pot + toCall) : 0;
    const spr = pot > 0 ? myStack / pot : 0;

    const diagnostics = {
      source: 'pro',
      botType: ctx.botType || 'PRO_GTO',
      decisionBranch: toCall > 0 ? 'facing_bet_postflop' : 'unopened_postflop',
      contextType: toCall > 0 ? 'facing_bet_postflop' : 'unopened_postflop',
      handName: handScore ? handScore.name : '—',
      madeHandClass: madeHandClass,
      drawClass: drawClass,
      boardTexture: texture.wet > 0.5 ? 'wet' : (texture.wet > 0.2 ? 'medium' : 'dry'),
      strength: strength,
      potOdds: potOdds,
      spr: spr,
      foldEquity: null,
      reason: ''
    };

    // Draw bonus: a flush draw has extra equity
    let effectiveStrength = strength;
    if (drawClass === 'combo_draw') effectiveStrength += 0.15;
    else if (drawClass === 'flush_draw' || drawClass === 'straight_draw') effectiveStrength += 0.10;
    effectiveStrength = Math.min(1, effectiveStrength);

    if (toCall === 0) {
      // No bet, we can check or bet
      if (effectiveStrength > 0.65) {
        const betSize = Math.round(pot * (texture.wet > 0.5 ? 0.65 : 0.5));
        diagnostics.reason = `value bet (${madeHandClass}, str=${effectiveStrength.toFixed(2)})`;
        return { action: 'bet', amount: Math.min(myStack, Math.max(bb, betSize)), diagnostics };
      }
      if (effectiveStrength > 0.42 && drawClass !== 'none') {
        const betSize = Math.round(pot * 0.5);
        diagnostics.reason = `semi-bluff bet (${drawClass})`;
        return { action: 'bet', amount: Math.min(myStack, Math.max(bb, betSize)), diagnostics };
      }
      diagnostics.reason = `check (${madeHandClass}, str=${effectiveStrength.toFixed(2)})`;
      return { action: 'check', amount: 0, diagnostics };
    }

    // Facing bet
    const requiredEquity = potOdds * 1.15;
    if (effectiveStrength > 0.72) {
      const raiseSize = Math.min(myStack, Math.round(pot * 2.5 + toCall));
      diagnostics.reason = `raise (${madeHandClass}, str=${effectiveStrength.toFixed(2)}, potOdds=${potOdds.toFixed(2)})`;
      return { action: 'raise', amount: raiseSize, diagnostics };
    }
    if (effectiveStrength > requiredEquity) {
      diagnostics.reason = `call (${madeHandClass}, str=${effectiveStrength.toFixed(2)} > req=${requiredEquity.toFixed(2)})`;
      return { action: 'call', amount: Math.min(toCall, myStack), diagnostics };
    }
    if (drawClass !== 'none' && effectiveStrength + 0.05 > requiredEquity && toCall < pot * 0.6) {
      diagnostics.reason = `draw call (${drawClass})`;
      return { action: 'call', amount: Math.min(toCall, myStack), diagnostics };
    }
    diagnostics.reason = `fold (${madeHandClass}, str=${effectiveStrength.toFixed(2)} < req=${requiredEquity.toFixed(2)})`;
    return { action: 'fold', amount: 0, diagnostics };
  }

  // ========================================================================
  // BLOCK 12 — DISPATCHER & PUBLIC FACADE
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
    try {
      return impl.decidePreflop(ctx);
    } catch (err) {
      return { action: 'fold', amount: 0, diagnostics: { error: 'decision_error', message: String(err && err.message || err) } };
    }
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
    analyze: analyzeBoardTexture
  };

  // ========================================================================
  // BLOCK 13 — EXPORT
  // ========================================================================

  const Bots = {
    CLASS_PRO,
    CLASS_VIP: null,
    TYPES,
    PROFILES,
    Texture: ProTexture,
    decidePreflop,
    decidePostflop,
    decide,
    assignBotTypes,
    assignBotsToSeats,
    preflopStrength,
    handInChart: function() { return false; },
    updateModel,
    updateMeta,
    getRead,
    reset,
    dumpStats,
    getProfile,
    getTypeMeta,
    updateOpponentStats,
    getOpponentType,
    getShrunk,
    detectContext,
    _internal: {
      calculateBubbleFactor,
      getStackDepth,
      getPositionMultiplier,
      getBlockerScore,
      getBlockerReason,
      equityRealization,
      shrink,
      applyStatModifier,
      NASH_PUSH_TABLE,
      BASE_OPEN_THRESHOLDS,
      POSITION_MULTIPLIERS,
      PRIORS,
      SHRINK_K,
      STAT_CONFIDENCE,
      opponentStats
    }
  };

  Bots.__level = 'top';
  Bots.__version = '3.0-preflop-full';
  Bots.__source = 'bots.js';

  window.Bots = Bots;
  window.__botsJsLoaded = true;

})();
