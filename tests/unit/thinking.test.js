import { describe, expect, it } from 'vitest';
import {
  appendThinkingAwareOutput, flushThinkingOutput, groupGeneratedTokens, thinkingStateFromPrompt,
} from '../../src/ui/thinking.js';

function view() {
  return {
    answerParts: [],
    answerText: '',
    thinkingParts: [],
    thinkingOpen: false,
    hasThinking: false,
    pendingChannel: false,
    channelBuffer: '',
    channelOwners: [],
    channelEntries: [],
    markerBuffer: '',
    markerOwners: [],
    tokenThinking: false,
    tokenStepsToMark: [],
    tokenStepsToUnmark: [],
    pendingBytes: false,
    pendingBytesThinking: false,
  };
}

function step(text, answerText = '', special = false, isEos = false, index = 0) {
  return {
    index,
    selected: { special, isEos, piece: { text } },
    answerText,
  };
}

describe('output con ragionamento separato', () => {
  it('inizializza dal marker finale del prompt solo se il template abilita il ragionamento', () => {
    const enabled = thinkingStateFromPrompt({
      thinking: true,
      fullText: '<|im_start|>assistant\n<think>\n',
    });
    expect(enabled).toMatchObject({ thinkingOpen: true, hasThinking: true });

    const noMarker = thinkingStateFromPrompt({ thinking: true, fullText: '<|im_start|>assistant\n' });
    const disabled = thinkingStateFromPrompt({ thinking: false, fullText: '<|im_start|>assistant\n<think>' });
    expect(noMarker).toMatchObject({ thinkingOpen: false, hasThinking: false });
    expect(disabled).toMatchObject({ thinkingOpen: false, hasThinking: false });

    const state = { ...view(), ...enabled };
    appendThinkingAwareOutput(state, step('Ragionamento iniziale.', 'Ragionamento iniziale.', false, false, 0));
    expect(state.thinkingParts.map((part) => part.text).join('')).toBe('Ragionamento iniziale.');
    expect(state.answerParts).toEqual([]);

    for (const initial of [noMarker, disabled]) {
      const ordinary = { ...view(), ...initial };
      appendThinkingAwareOutput(ordinary, step('Risposta.', 'Risposta.', false, false, 0));
      expect(ordinary.answerParts.map((part) => part.text).join('')).toBe('Risposta.');
      expect(ordinary.thinkingParts).toEqual([]);
    }
  });

  it('separa i marker Qwen dal testo di ragionamento e dalla risposta', () => {
    const state = view();
    appendThinkingAwareOutput(state, step('<think>', '', true, false, 0));
    appendThinkingAwareOutput(state, step('Considero i dati. ', 'Considero i dati. ', false, false, 1));
    appendThinkingAwareOutput(state, step('</think>', 'Considero i dati. ', true, false, 2));
    appendThinkingAwareOutput(state, step('La risposta è 42.', 'Considero i dati. La risposta è 42.', false, false, 3));

    expect(state.thinkingParts.map((part) => part.text).join('')).toBe('Considero i dati. ');
    expect(state.answerParts.map((part) => part.text).join('')).toBe('La risposta è 42.');
    expect(state.hasThinking).toBe(true);
    expect(state.thinkingOpen).toBe(false);
  });

  it('separa il canale thought Gemma dal canale final', () => {
    const state = view();
    appendThinkingAwareOutput(state, step('<|channel>', '', true, false, 0));
    appendThinkingAwareOutput(state, step('thought\n', 'thought\n', false, false, 1));
    const thoughtMarkerTokens = state.tokenStepsToMark.slice();
    appendThinkingAwareOutput(state, step('Valuto la domanda. ', 'thought\nValuto la domanda. ', false, false, 2));
    appendThinkingAwareOutput(state, step('<channel|>', 'thought\nValuto la domanda. ', true, false, 3));
    appendThinkingAwareOutput(state, step('<|channel>', 'thought\nValuto la domanda. ', true, false, 4));
    appendThinkingAwareOutput(state, step('final\n', 'thought\nValuto la domanda. final\n', false, false, 5));
    appendThinkingAwareOutput(state, step('Ecco il risultato.', 'thought\nValuto la domanda. final\nEcco il risultato.', false, false, 6));

    expect(state.thinkingParts.map((part) => part.text).join('')).toBe('Valuto la domanda. ');
    expect(state.answerParts.map((part) => part.text).join('')).toBe('Ecco il risultato.');
    expect(state.thinkingOpen).toBe(false);
    expect(thoughtMarkerTokens).toEqual([0, 1]);
  });

  it('riconosce marker Qwen come testo ordinario anche quando sono divisi tra token', () => {
    const state = view();
    appendThinkingAwareOutput(state, step('<th', '<th', false, false, 0));
    expect(state.answerParts).toEqual([]);
    appendThinkingAwareOutput(state, step('ink>', '<think>', false, false, 1));
    const openingMarkerTokens = state.tokenStepsToMark.slice();
    appendThinkingAwareOutput(state, step('Ragiono. ', '<think>Ragiono. ', false, false, 2));
    appendThinkingAwareOutput(state, step('</th', '<think>Ragiono. </th', false, false, 3));
    appendThinkingAwareOutput(state, step('ink>', '<think>Ragiono. </think>', false, false, 4));
    appendThinkingAwareOutput(state, step('Risposta.', '<think>Ragiono. </think>Risposta.', false, false, 5));

    expect(state.thinkingParts.map((part) => part.text).join('')).toBe('Ragiono. ');
    expect(state.answerParts.map((part) => part.text).join('')).toBe('Risposta.');
    expect(openingMarkerTokens).toEqual([0, 1]);
  });

  it('mostra come testo ordinario un marker incompleto a fine generazione', () => {
    const state = view();
    appendThinkingAwareOutput(state, step('<thi', '<thi', false, false, 0));
    flushThinkingOutput(state);

    expect(state.answerParts.map((part) => part.text).join('')).toBe('<thi');
    expect(state.markerBuffer).toBe('');
  });

  it('riconosce anche i marker di canale Gemma come testo ordinario e diviso', () => {
    const state = view();
    appendThinkingAwareOutput(state, step('<|cha', '<|cha', false, false, 0));
    appendThinkingAwareOutput(state, step('nnel>tho', '<|channel>tho', false, false, 1));
    appendThinkingAwareOutput(state, step('ught\nPassaggio. ', '<|channel>thought\nPassaggio. ', false, false, 2));
    const thoughtMarkerTokens = state.tokenStepsToMark.slice();
    appendThinkingAwareOutput(state, step('<channel|>', '<|channel>thought\nPassaggio. <channel|>', false, false, 3));
    appendThinkingAwareOutput(state, step('Risultato.', '<|channel>thought\nPassaggio. <channel|>Risultato.', false, false, 4));

    expect(state.thinkingParts.map((part) => part.text).join('')).toBe('Passaggio. ');
    expect(state.answerParts.map((part) => part.text).join('')).toBe('Risultato.');
    expect(thoughtMarkerTokens).toEqual([0, 1, 2]);
    expect(state.thinkingOpen).toBe(false);
  });

  it('riclassifica i token del canale final se thought non ha un marker di chiusura separato', () => {
    const state = view();
    appendThinkingAwareOutput(state, step('<|channel>', '', true, false, 0));
    appendThinkingAwareOutput(state, step('thought\n', 'thought\n', false, false, 1));
    appendThinkingAwareOutput(state, step('Ragiono.', 'thought\nRagiono.', false, false, 2));
    appendThinkingAwareOutput(state, step('<|channel>', 'thought\nRagiono.', true, false, 3));
    appendThinkingAwareOutput(state, step('final\n', 'thought\nRagiono.final\n', false, false, 4));

    expect(state.thinkingOpen).toBe(false);
    expect(state.tokenStepsToUnmark).toEqual([3, 4]);
  });

  it('mantiene controlli e testo ordinario visibili fuori da un blocco di ragionamento', () => {
    const state = view();
    appendThinkingAwareOutput(state, step('Ciao.', 'Ciao.', false, false, 0));
    appendThinkingAwareOutput(state, step('<eos>', 'Ciao.', true, true, 1));

    expect(state.answerParts).toEqual([
      { text: 'Ciao.' },
      { control: true, eos: true, text: '<eos>' },
    ]);
    expect(state.thinkingParts).toEqual([]);
    expect(state.hasThinking).toBe(false);
  });

  it('raggruppa le tessere di ragionamento senza eliminarle o riordinarle', () => {
    const tokens = [
      { id: 1, thinking: false },
      { id: 2, thinking: true },
      { id: 3, thinking: true },
      { id: 4, thinking: false },
      { id: 5, thinking: true },
    ];
    const groups = groupGeneratedTokens(tokens);
    expect(groups.map((group) => [group.thinking, group.tokens.map((token) => token.id)])).toEqual([
      [false, [1]], [true, [2, 3]], [false, [4]], [true, [5]],
    ]);
    expect(groups.flatMap((group) => group.tokens)).toEqual(tokens);
  });
});
