import { REPLACEMENT } from '../core/tokens.js';

const MARKERS = ['<think>', '</think>', '<|channel>', '<channel|>'];
const CHANNELS = ['thought', 'analysis', 'final'];

export function thinkingStateFromPrompt(input, requestedThinking = input?.thinking) {
  const initial = { thinkingOpen: false, hasThinking: false, pendingChannel: false };
  if (requestedThinking === false) return initial;

  const prompt = String(input?.fullText ?? '');
  if (/<think>\s*$/i.test(prompt) || /<\|channel>(?:thought|analysis)\s*$/i.test(prompt)) {
    return { ...initial, thinkingOpen: true, hasThinking: true };
  }
  if (/<\|channel>\s*$/i.test(prompt)) return { ...initial, pendingChannel: true };
  return initial;
}

function addOwner(owners, owner) {
  if (Number.isInteger(owner) && !owners.includes(owner)) owners.push(owner);
}

function markThinkingTokens(view, owners) {
  for (const owner of owners) {
    addOwner(view.tokenStepsToMark, owner);
    if (owner === view.currentStepIndex) view.tokenThinking = true;
  }
}

function unmarkThinkingTokens(view, owners) {
  for (const owner of owners) {
    addOwner(view.tokenStepsToUnmark, owner);
    if (owner === view.currentStepIndex) view.tokenThinking = false;
  }
}

function appendText(view, text, thinking = view.thinkingOpen, owners = []) {
  if (!text) return;
  (thinking ? view.thinkingParts : view.answerParts).push({ text });
  if (thinking) {
    for (const owner of owners) {
      if (owner === view.currentStepIndex) view.tokenThinking = true;
    }
  }
}

function markerText(view, chars, owners, thinking = view.thinkingOpen) {
  appendText(view, chars.map((entry) => entry.char).join(''), thinking, owners);
  view.markerBuffer = '';
  view.markerOwners = [];
}

function appendChannelText(view, chars) {
  view.channelEntries.push(...chars);
  view.channelBuffer = view.channelEntries.map((entry) => entry.char).join('');
  for (const entry of chars) addOwner(view.channelOwners, entry.owner);

  const leading = view.channelBuffer.match(/^\s*/)?.[0] ?? '';
  const label = view.channelBuffer.slice(leading.length);
  const channel = CHANNELS.find((name) => label === name || label.startsWith(`${name}\n`) || label.startsWith(`${name}\r`));
  if (channel) {
    const metadataLength = leading.length + channel.length;
    const consumed = view.channelBuffer.slice(metadataLength).match(/^[ \t]*\r?\n/)?.[0].length ?? 0;
    const owners = view.channelOwners.slice();
    const rest = view.channelEntries.slice(metadataLength + consumed);
    if (channel !== 'final') {
      view.thinkingOpen = true;
      view.hasThinking = true;
      markThinkingTokens(view, owners);
    } else {
      view.thinkingOpen = false;
      unmarkThinkingTokens(view, owners);
    }
    view.pendingChannel = false;
    view.channelBuffer = '';
    view.channelOwners = [];
    view.channelEntries = [];
    if (rest.length) scanText(view, rest);
    return;
  }

  if (CHANNELS.some((name) => name.startsWith(label))) return;

  const buffered = view.channelBuffer;
  const owners = view.channelOwners.slice();
  view.pendingChannel = false;
  view.channelBuffer = '';
  view.channelOwners = [];
  view.channelEntries = [];
  appendText(view, buffered, view.thinkingOpen, owners);
  return;
}

function scanText(view, chars) {
  if (view.markerBuffer) {
    const pending = [...view.markerBuffer].map((char, i) => ({ char, owner: view.markerOwners[i] }));
    chars = pending.concat(chars);
    view.markerBuffer = '';
    view.markerOwners = [];
  }
  let pos = 0;
  let text = [];
  let textOwners = [];
  let textMode = view.thinkingOpen;
  const flushText = () => {
    if (text.length) appendText(view, text.join(''), textMode, textOwners);
    text = [];
    textOwners = [];
  };

  while (pos < chars.length) {
    if (view.pendingChannel) {
      flushText();
      appendChannelText(view, chars.slice(pos));
      return;
    }

    const tail = chars.slice(pos).map((entry) => entry.char).join('');
    const complete = MARKERS.find((marker) => tail.startsWith(marker));
    if (complete) {
      flushText();
      const owners = [];
      chars.slice(pos, pos + complete.length).forEach((entry) => addOwner(owners, entry.owner));
      if (complete === '<|channel>') {
        view.pendingChannel = true;
        view.channelOwners = owners;
        view.channelBuffer = '';
        view.channelEntries = [];
      } else if (complete === '<channel|>') {
        if (view.thinkingOpen) markThinkingTokens(view, owners);
        view.thinkingOpen = false;
      } else if (complete === '<think>') {
        view.thinkingOpen = true;
        view.hasThinking = true;
        markThinkingTokens(view, owners);
      } else {
        if (view.thinkingOpen) markThinkingTokens(view, owners);
        view.thinkingOpen = false;
      }
      pos += complete.length;
      textMode = view.thinkingOpen;
      continue;
    }

    if (MARKERS.some((marker) => marker.startsWith(tail))) {
      flushText();
      view.markerBuffer = tail;
      view.markerOwners = chars.slice(pos).map((entry) => entry.owner);
      return;
    }

    const entry = chars[pos];
    if (text.length && textMode !== view.thinkingOpen) flushText();
    textMode = view.thinkingOpen;
    text.push(entry.char);
    addOwner(textOwners, entry.owner);
    pos += 1;
  }
  flushText();
}

function flushPendingText(view) {
  if (view.markerBuffer) {
    const chars = [...view.markerBuffer].map((char, i) => ({ char, owner: view.markerOwners[i] }));
    markerText(view, chars, view.markerOwners);
  }
  if (view.pendingChannel) {
    appendText(view, view.channelBuffer, view.thinkingOpen, view.channelOwners);
    view.pendingChannel = false;
    view.channelBuffer = '';
    view.channelOwners = [];
    view.channelEntries = [];
  }
}

export function flushThinkingOutput(view) {
  flushPendingText(view);
}

/**
 * Appends one generated token to the appropriate visible stream.
 * Thinking markers are structural whether the tokenizer marks them special or not.
 */
export function appendThinkingAwareOutput(view, step) {
  const selected = step.selected;
  const piece = selected.piece ?? {};
  const marker = String(piece.text ?? '').trim().toLowerCase();
  view.currentStepIndex = step.index;
  view.tokenThinking = Boolean(view.thinkingOpen);
  view.tokenStepsToMark = [];
  view.tokenStepsToUnmark = [];

  if (selected.special) {
    flushPendingText(view);
    if (marker === '<think>') {
      view.thinkingOpen = true;
      view.hasThinking = true;
      markThinkingTokens(view, [step.index]);
    } else if (marker === '</think>') {
      if (view.thinkingOpen) markThinkingTokens(view, [step.index]);
      view.thinkingOpen = false;
    } else if (marker === '<|channel>') {
      view.pendingChannel = true;
      view.channelBuffer = '';
      view.channelOwners = [step.index];
      view.channelEntries = [];
    } else if (marker === '<|channel>thought' || marker === '<|channel>analysis') {
      view.thinkingOpen = true;
      view.hasThinking = true;
      markThinkingTokens(view, [step.index]);
    } else if (marker === '<|channel>final' || marker === '<channel|>') {
      if (marker === '<channel|>' && view.thinkingOpen) markThinkingTokens(view, [step.index]);
      if (marker === '<|channel>final') unmarkThinkingTokens(view, [step.index]);
      view.thinkingOpen = false;
    } else {
      (view.thinkingOpen ? view.thinkingParts : view.answerParts).push({
        control: true,
        eos: selected.isEos,
        text: piece.text ?? '',
      });
    }
    return;
  }

  const decoded = step.answerText ?? '';
  let next = decoded;
  while (next.endsWith(REPLACEMENT)) next = next.slice(0, -1);
  view.pendingBytes = next.length !== decoded.length;
  const delta = next.startsWith(view.answerText) ? next.slice(view.answerText.length) : next;
  view.answerText = next;
  scanText(view, [...delta].map((char) => ({ char, owner: step.index })));
  view.pendingBytesThinking = view.pendingBytes && view.thinkingOpen;
}

export function groupGeneratedTokens(tokens) {
  const groups = [];
  for (const token of tokens) {
    const thinking = Boolean(token.thinking);
    const previous = groups.at(-1);
    if (thinking && previous?.thinking) {
      previous.tokens.push(token);
    } else {
      groups.push({ thinking, tokens: [token] });
    }
  }
  return groups;
}
