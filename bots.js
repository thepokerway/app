(function() {
  'use strict';

  const R = Engine.RANK_VALUE;
  const CLASS_PRO = 'PRO';

  const TYPES = {
    PRO_GTO:    { key: 'PRO_GTO',    cls: CLASS_PRO },
    PRO_SOLVER: { key: 'PRO_SOLVER', cls: CLASS_PRO },
    PRO_ICM:    { key: 'PRO_ICM',    cls: CLASS_PRO },
    PRO_LAG:    { key: 'PRO_LAG',    cls: CLASS_PRO }
  };

  const PROFILES = {
    PRO_GTO:    { aggression: 0.68, looseness: 0.42, bluff: 0.22, callThreshold: 0.55 },
    PRO_SOLVER: { aggression: 0.78, looseness: 0.55, bluff: 0.34, callThreshold: 0.45 },
    PRO_ICM:    { aggression: 0.72, looseness: 0.45, bluff: 0.28, callThreshold: 0.52 },
    PRO_LAG:    { aggression: 0.92, looseness: 0.72, bluff: 0.48, callThreshold: 0.32 }
  };

  const CHARTS = {
    OPEN_UTG:  ['22+','A2s+','K7s+','Q8s+','J8s+','T8s+','97s+','86s+','75s+','64s+','54s','ATo+','KJo+','QJo','JTo'],
    OPEN_UTG1: ['22+','A2s+','K6s+','Q8s+','J8s+','T8s+','97s+','86s+','75s+','64s+','54s','ATo+','KTo+','QTo+','JTo'],
    OPEN_MP:   ['22+','A2s+','K5s+','Q7s+','J7s+','T7s+','96s+','85s+','75s+','64s+','53s+','43s','A9o+','KTo+','QTo+','JTo','T9o'],
    OPEN_HJ:   ['22+','A2s+','K3s+','Q6s+','J7s+','T7s+','96s+','85s+','74s+','63s+','52s+','43s','A7o+','KTo+','QTo+','JTo','T9o','98o'],
    OPEN_CO:   ['22+','A2s+','K2s+','Q5s+','J6s+','T6s+','95s+','84s+','73s+','63s+','52s+','42s+','32s','A4o+','K9o+','Q9o+','J9o+','T9o','98o','87o'],
    OPEN_BTN:  ['22+','A2s+','K2s+','Q2s+','J3s+','T5s+','94s+','83s+','72s+','62s+','52s+','42s+','32s','A2o+','K5o+','Q7o+','J8o+','T8o+','98o','87o','76o','65o'],
    OPEN_SB:   ['22+','A2s+','K2s+','Q2s+','J2s+','T4s+','93s+','82s+','72s+','62s+','52s+','42s+','32s','A2o+','K4o+','Q6o+','J7o+','T8o+','98o','87o','76o','65o','54o'],
    THREE_BET_VS_UTG: ['QQ+','AKs','AKo','AQs','JJ','TT'],
    THREE_BET_VS_MP:  ['TT+','AKs','AKo','AQs','AQo','AJs','KQs','99'],
    THREE_BET_VS_CO:  ['99+','AQs+','AKo','AQo','AJs+','ATs','KQs','KJs','QJs','88'],
    THREE_BET_VS_BTN: ['77+','ATs+','AJo+','KTs+','KQo','QTs+','JTs','T9s','98s','A5s','A4s','66'],
    THREE_BET_VS_SB:  ['66+','A9s+','ATo+','KTs+','KJo+','QTs+','QJo','JTs','T9s','98s','A5s+'],
    CALL_RAISE: ['22-99','A2s+','K9s+','Q9s+','J9s+','T8s+','98s','87s','76s','65s','54s','ATo+','KJo+','QJo','JTo','A9o','KTo','QTo'],
    CALL_3BET:  ['JJ','TT','99','88','AQs','AJs','KQs','AQo','ATs'],
    SHOVE_10_15BB: ['22+','A2s+','K5s+','Q7s+','J8s+','T8s+','98s','87s','76s','A7o+','K9o+','Q9o+','J9o+','T9o'],
    SHOVE_5_10BB:  ['22+','A2s+','K2s+','Q4s+','J6s+','T7s+','96s+','85s+','74s+','64s+','54s','A2o+','K7o+','Q8o+','J9o+','T9o','98o'],
    CALL_SHOVE: ['77+','AQs+','AKo','TT','JJ','QQ','KK','AA']
  };

  const chartCache = {};

  function normalizeHand(c1, c2) {
    const r1 = c1.rank, r2 = c2.rank;
    const v1 = R[r1], v2 = R[r2];
    if (v1 === v2) return r1 + r2;
    const hi = v1 > v2 ? r1 : r2;
    const lo = v1 > v2 ? r2 : r1;
    const suited = c1.suit === c2.suit ? 's' : 'o';
    return hi + lo + suited;
  }

  function rankFromValue(v) {
    for (const r of Engine.RANKS) if (R[r] === v) return r;
    return '?';
  }

  function expandChart(entries) {
    const set = new Set();
    for (const entry of entries) {
      let m;
      if ((m = entry.match(/^([2-9TJQKA])\1-([2-9TJQKA])\2$/))) {
        const from = R[m[1]], to = R[m[2]];
        for (let v = from; v <= to; v++) { const r = rankFromValue(v); set.add(r + r); }
        continue;
      }
      if ((m = entry.match(/^([2-9TJQKA])\1\+$/))) {
        const from = R[m[1]];
        for (let v = from; v <= 14; v++) { const r = rankFromValue(v); set.add(r + r); }
        continue;
      }
      if ((m = entry.match(/^([2-9TJQKA])\1$/))) { set.add(m[1] + m[1]); continue; }
      if ((m = entry.match(/^([2-9TJQKA])([2-9TJQKA])([so])\+$/))) {
        const hi = m[1], lo = m[2], suit = m[3];
        const hiV = R[hi], loV = R[lo];
        for (let v = loV; v < hiV; v++) { const r = rankFromValue(v); set.add(hi + r + suit); }
        continue;
      }
      if ((m = entry.match(/^([2-9TJQKA])([2-9TJQKA])([so])$/))) { set.add(m[1] + m[2] + m[3]); continue; }
      if ((m = entry.match(/^([2-9TJQKA])([2-9TJQKA])([so])-([2-9TJQKA])([2-9TJQKA])([so])$/))) {
        const hi = m[1], loFrom = R[m[2]], loTo = R[m[4]], suit = m[3];
        const hiV = R[hi];
        for (let v = loFrom; v <= loTo; v++) {
          if (v >= hiV) continue;
          const r = rankFromValue(v); set.add(hi + r + suit);
        }
        continue;
      }
    }
    return set;
  }

  function getChart(name) {
    if (!chartCache[name]) chartCache[name] = expandChart(CHARTS[name] || []);
    return chartCache[name];
  }

  function handInChart(card1, card2, chartName) {
    if (!CHARTS[chartName]) return false;
    return getChart(chartName).has(normalizeHand(card1, card2));
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

  function preflopStrength(card1, card2) {
    const v1 = R[card1.rank], v2 = R[card2.rank];
    const hi = Math.max(v1, v2), lo = Math.min(v1, v2);
    const suited = card1.suit === card2.suit;
    if (v1 === v2) return 0.5 + (hi - 2) / 12 * 0.5;
    let s = (hi - 2) / 12 * 0.55 + (lo - 2) / 12 * 0.25;
    if (suited) s += 0.08;
    if (hi === 14) s += 0.05;
    if (Math.abs(hi - lo) === 1) s += 0.04;
    if (Math.abs(hi - lo) === 2) s += 0.02;
    return Math.min(1, Math.max(0, s));
  }

  function postflopStrength(handScore) {
    if (!handScore) return 0;
    const base = [0.15,0.35,0.55,0.70,0.78,0.85,0.92,0.97,0.99,1.00][handScore.category] || 0;
    const rank = handScore.ranks[0] || 0;
    return Math.min(1, base + (rank / 14) * 0.05);
  }

  function openChartFor(position) {
    switch (position) {
      case 'UTG':  return 'OPEN_UTG';
      case 'UTG1': return 'OPEN_UTG1';
      case 'LJ':   return 'OPEN_MP';
      case 'HJ':   return 'OPEN_HJ';
      case 'CO':   return 'OPEN_CO';
      case 'BTN':  return 'OPEN_BTN';
      case 'SB':   return 'OPEN_SB';
      case 'BB':   return 'OPEN_BTN';
      default:     return 'OPEN_BTN';
    }
  }

  function threeBetChartFor(raiserPos) {
    switch (raiserPos) {
      case 'UTG': case 'UTG1': return 'THREE_BET_VS_UTG';
      case 'LJ':  case 'HJ':   return 'THREE_BET_VS_MP';
      case 'CO':               return 'THREE_BET_VS_CO';
      case 'BTN':              return 'THREE_BET_VS_BTN';
      case 'SB':               return 'THREE_BET_VS_SB';
      case 'BB':               return 'THREE_BET_VS_SB';
      default:                 return 'THREE_BET_VS_CO';
    }
  }

  function makeDiagnostics(base) {
    return {
      source: 'pro',
      strength: base.strength,
      preflopStrength: base.preflopStrength,
      postflopStrength: base.postflopStrength,
      handName: base.handName,
      chart: base.chart,
      threshold: base.threshold,
      reason: base.reason
    };
  }

  function decidePreflop(ctx) {
    const profile = PROFILES[ctx.botType];
    if (!profile) return { action: 'fold', amount: 0, diagnostics: makeDiagnostics({ reason: 'no profile' }) };

    const card1 = ctx.card1, card2 = ctx.card2;
    const position = ctx.position;
    const myStack = ctx.myStack;
    const bb = ctx.bb;
    const toCall = ctx.toCall;
    const pot = ctx.pot;
    const numActive = ctx.numActive;

    const strength = preflopStrength(card1, card2);
    const stackBB = myStack / bb;

    const diagBase = {
      strength: strength,
      preflopStrength: strength,
      postflopStrength: null,
      handName: '—',
      chart: null,
      threshold: null,
      reason: ''
    };

    if (stackBB <= 15 && !ctx.facingRaise) {
      if (stackBB <= 10 && handInChart(card1, card2, 'SHOVE_5_10BB')) {
        return { action: 'allin', amount: myStack, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'SHOVE_5_10BB', threshold: 'short stack shove', reason: 'short stack shove chart' })) };
      }
      if (stackBB > 10 && stackBB <= 15 && handInChart(card1, card2, 'SHOVE_10_15BB')) {
        return { action: 'allin', amount: myStack, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'SHOVE_10_15BB', threshold: 'mid stack shove', reason: 'mid stack shove chart' })) };
      }
      if (toCall === 0) return { action: 'check', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'short stack check', reason: 'short stack, not in shove chart, check' })) };
      return { action: 'fold', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'short stack fold', reason: 'short stack, not in shove chart, fold' })) };
    }

    if (ctx.facingRaise && ctx.facingAllin) {
      if (handInChart(card1, card2, 'CALL_SHOVE') || (profile.looseness > 0.7 && strength > 0.55)) {
        return { action: 'call', amount: toCall, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'CALL_SHOVE', threshold: 'call shove', reason: 'in call shove chart' })) };
      }
      return { action: 'fold', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'CALL_SHOVE', threshold: 'fold vs shove', reason: 'not in call shove chart, fold' })) };
    }

    if (ctx.facing3Bet) {
      if (handInChart(card1, card2, 'CALL_3BET') || (profile.looseness > 0.6 && strength > 0.6)) {
        if (profile.aggression > 0.7 && strength > 0.8 && Math.random() < 0.5) {
          const size = Math.min(myStack, Math.round((pot + toCall) * 2.2));
          return { action: 'raise', amount: size, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'CALL_3BET', threshold: 're-raise', reason: 'strong vs 3bet, re-raise' })) };
        }
        return { action: 'call', amount: toCall, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'CALL_3BET', threshold: 'call 3bet', reason: 'in call 3bet chart' })) };
      }
      return { action: 'fold', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'CALL_3BET', threshold: 'fold vs 3bet', reason: 'not in call 3bet, fold' })) };
    }

    if (ctx.facingRaise) {
      const chart3bet = threeBetChartFor(ctx.raiserPosition);
      if (handInChart(card1, card2, chart3bet)) {
        const size = Math.min(myStack, Math.round((pot + toCall) * 3));
        return { action: 'raise', amount: size, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: chart3bet, threshold: '3bet', reason: 'in 3bet chart' })) };
      }
      if (handInChart(card1, card2, 'CALL_RAISE')) {
        return { action: 'call', amount: toCall, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'CALL_RAISE', threshold: 'call raise', reason: 'in call raise chart' })) };
      }
      if (profile.looseness > 0.6 && strength > 0.35) {
        return { action: 'call', amount: toCall, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'CALL_RAISE', threshold: 'loose call', reason: 'loose call' })) };
      }
      if (profile.bluff > 0.4 && strength > 0.3 && Math.random() < profile.bluff * 0.3) {
        const size = Math.min(myStack, Math.round((pot + toCall) * 3));
        return { action: 'raise', amount: size, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: chart3bet, threshold: 'bluff 3bet', reason: 'bluff 3bet' })) };
      }
      return { action: 'fold', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: 'CALL_RAISE', threshold: 'fold vs raise', reason: 'not in chart, fold' })) };
    }

    const openChart = openChartFor(position);
    if (handInChart(card1, card2, openChart)) {
      const openSize = Math.round(bb * (numActive > 5 ? 3 : 2.5));
      return { action: 'raise', amount: Math.min(myStack, openSize), diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: openChart, threshold: 'open', reason: 'in open chart, raise' })) };
    }

    if (profile.looseness > 0.7 && strength > 0.25 && Math.random() < profile.bluff * 0.5) {
      const openSize = Math.round(bb * 2.5);
      return { action: 'raise', amount: Math.min(myStack, openSize), diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: openChart, threshold: 'loose bluff', reason: 'loose bluff open' })) };
    }
    if (toCall === 0) return { action: 'check', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: openChart, threshold: 'check', reason: 'not in open chart, check' })) };
    return { action: 'fold', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { chart: openChart, threshold: 'fold', reason: 'not in open chart, fold' })) };
  }

  function decidePostflop(ctx) {
    const profile = PROFILES[ctx.botType];
    if (!profile) return { action: 'fold', amount: 0, diagnostics: makeDiagnostics({ reason: 'no profile' }) };

    const card1 = ctx.card1, card2 = ctx.card2;
    const board = ctx.board || [];
    const toCall = ctx.toCall || 0;
    const pot = ctx.pot || 0;
    const myStack = ctx.myStack || 0;
    const bb = ctx.bb || 1;
    const street = ctx.street;

    const allCards = [card1, card2, ...board];
    const handScore = allCards.length >= 5 ? Engine.evaluateHand(allCards) : null;
    const strength = handScore ? postflopStrength(handScore) : preflopStrength(card1, card2);
    const streetBonus = street === 'river' ? 0.05 : 0;

    const diagBase = {
      strength: strength,
      preflopStrength: null,
      postflopStrength: strength,
      handName: handScore ? handScore.name : '—',
      chart: null,
      threshold: null,
      reason: ''
    };

    if (toCall === 0) {
      if (profile.aggression > 0.6 && strength > 0.85 && street === 'flop') {
        if (Math.random() < 0.35) {
          return { action: 'check', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'trap', reason: 'strong, trap check' })) };
        }
      }
      if (strength + streetBonus > 0.65) {
        const betSize = Math.round(pot * 0.5);
        return { action: 'bet', amount: Math.min(myStack, Math.max(bb, betSize)), diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'strong bet', reason: 'strong hand, bet' })) };
      }
      if (strength > 0.4 && Math.random() < profile.aggression * 0.5) {
        const betSize = Math.round(pot * 0.5);
        return { action: 'bet', amount: Math.min(myStack, Math.max(bb, betSize)), diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'medium bet', reason: 'medium, aggressive bet' })) };
      }
      if (Math.random() < profile.bluff * 0.25) {
        const betSize = Math.round(pot * 0.6);
        return { action: 'bet', amount: Math.min(myStack, Math.max(bb, betSize)), diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'bluff bet', reason: 'bluff bet' })) };
      }
      return { action: 'check', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'check', reason: 'weak, check' })) };
    }

    const potOdds = toCall / (pot + toCall);

    if (strength + streetBonus > 0.75) {
      if (profile.aggression > 0.5 && Math.random() < profile.aggression) {
        const raiseSize = Math.min(myStack, Math.round((pot + toCall) * 2.2));
        return { action: 'raise', amount: raiseSize, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'very strong raise', reason: 'very strong, raise' })) };
      }
      return { action: 'call', amount: toCall, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'very strong call', reason: 'very strong, call' })) };
    }

    if (strength > profile.callThreshold && strength > potOdds * 1.5) {
      return { action: 'call', amount: toCall, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'call', reason: 'above threshold, call' })) };
    }

    if (strength > 0.3 && Math.random() < profile.bluff * 0.15) {
      const raiseSize = Math.min(myStack, Math.round((pot + toCall) * 2.5));
      return { action: 'raise', amount: raiseSize, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'bluff raise', reason: 'bluff raise' })) };
    }

    return { action: 'fold', amount: 0, diagnostics: makeDiagnostics(Object.assign({}, diagBase, { threshold: 'fold', reason: 'below threshold, fold' })) };
  }

  function decide(ctx) {
    if (!ctx || !ctx.botType) return { action: 'fold', amount: 0, diagnostics: makeDiagnostics({ reason: 'invalid ctx' }) };
    return ctx.street === 'preflop' ? decidePreflop(ctx) : decidePostflop(ctx);
  }

  function getProfile(botType) { return PROFILES[botType] || null; }
  function getTypeMeta(botType) { return TYPES[botType] || null; }

  function updateModel() {}
  function updateMeta() {}
  function getRead() { return null; }
  function reset() {}
  function dumpStats() { return []; }

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
    postflopStrength,
    handInChart,
    updateModel,
    updateMeta,
    getRead,
    reset,
    dumpStats,
    getProfile,
    getTypeMeta
  };

  Bots.__level = 'top';
  Bots.__version = '1.0';
  Bots.__source = 'bots.js';

  window.Bots = Bots;
  window.__botsJsLoaded = true;

})();
