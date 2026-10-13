const $ = (id) => document.getElementById(id);
const state = { job: null, results: [], selected: null, context: null, messages: [], version: 0 };
const dayFormat = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
const timeFormat = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' });
const date = (value) => dayFormat.format(new Date(value));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Something went wrong. Try again.');
  return data;
}

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = false;
}

function clearConversation() {
  state.version++;
  state.selected = null;
  state.context = null;
  $('conversation').hidden = true;
  $('conversation-empty').hidden = false;
  $('conversation-empty').querySelector('h2').textContent = 'Select a result.';
  document.querySelector('.workspace').classList.remove('reading');
}

function renderResults() {
  const list = $('results');
  list.replaceChildren();
  $('result-count').textContent = String(state.results.length).padStart(2, '0');
  if (!state.results.length) {
    list.append(element('div', 'list-empty', 'No matching conversations.\nTry a different detail or phrase.'));
    return;
  }
  for (const hit of state.results) {
    const button = element('button', 'result');
    button.type = 'button';
    button.dataset.index = hit.index;
    button.setAttribute('aria-pressed', String(state.selected === hit.index));
    const top = element('div', 'result-top');
    top.append(element('span', 'result-title', hit.title), element('span', 'result-date', date(hit.start)));
    const bottom = element('div', 'result-bottom');
    bottom.append(element('span', '', `${hit.message_count} messages`), element('span', '', `${Math.round(hit.probability * 100)}% match`));
    button.append(top, element('div', 'result-preview', hit.preview), bottom);
    button.addEventListener('click', () => openResult(hit.index));
    list.append(button);
  }
}

async function search(query, existingJob = null) {
  $('error').hidden = true;
  $('warnings').hidden = true;
  $('search-button').disabled = true;
  $('retrieval-text').disabled = true;
  $('summary-weight').disabled = true;
  $('query').readOnly = true;
  $('search-button').firstChild.textContent = 'Searching ';
  $('status').textContent = 'Planning your search…';
  const started = Date.now();
  let connected = false;
  try {
    const job = existingJob ? { id: existingJob } : await api('/api/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query,
        retrieval_text: $('retrieval-text').value, summary_weight: Number($('summary-weight').value) })
    });
    state.job = job.id;
    connected = true;
    sessionStorage.setItem('message-search-job', job.id);
    clearConversation();
    state.results = [];
    $('results').replaceChildren(element('div', 'list-empty', 'Looking through your conversations…'));
    $('result-count').textContent = '…';
    while (true) {
      const result = await api(`/api/search/${job.id}`);
      $('query').value = result.query;
      $('retrieval-text').value = result.retrieval_text;
      $('summary-weight').value = result.summary_weight;
      $('weight-control').hidden = result.retrieval_text !== 'both';
      $('elapsed').textContent = `${Math.floor((Date.now() - started) / 1000)}s`;
      if (result.status === 'error') throw new Error(result.error);
      if (result.status === 'done') {
        state.results = result.results;
        renderResults();
        $('status').textContent = `${result.results.length} matching conversation${result.results.length === 1 ? '' : 's'}${result.complete ? '' : ' · Some passages could not be checked'}`;
        $('warning-list').replaceChildren(...result.warnings.map((text) => element('li', '', text)));
        $('warnings').hidden = !result.warnings.length;
        break;
      }
      const completed = result.completed_stages;
      $('status').textContent = completed.includes('fuse') ? 'Checking the closest matches…'
        : completed.includes('plan') ? 'Searching your conversations…' : 'Planning your search…';
      await sleep(1000);
    }
  } catch (error) {
    showError(error.message);
    $('status').textContent = 'Search could not finish.';
    if (connected) {
      $('results').replaceChildren(element('div', 'list-empty', 'Search interrupted. Try your query again.'));
      $('result-count').textContent = '—';
    }
    // Keep a running job available for recovery after a transient connection failure.
    if (!connected) sessionStorage.removeItem('message-search-job');
  } finally {
    $('search-button').disabled = false;
    $('retrieval-text').disabled = false;
    $('summary-weight').disabled = false;
    $('query').readOnly = false;
    $('search-button').firstChild.textContent = 'Search ';
  }
}

function renderMessages() {
  const container = $('messages');
  const nodes = [];
  const matched = new Set(state.context.matched_ids);
  let previousDay = null;
  for (const message of state.messages) {
    const day = message.timestamp.slice(0, 10);
    if (day !== previousDay) nodes.push(element('div', 'day', date(message.timestamp)));
    previousDay = day;
    const mine = message.sender.toLowerCase() === 'me';
    const event = message.record_type === 'event';
    const row = element('article', `message${mine ? ' mine' : ''}${event ? ' event' : ''}${matched.has(message.id) ? ' matched' : ''}`);
    row.dataset.messageId = message.id;
    if (!event) row.append(element('div', 'message-meta', `${mine ? 'You' : message.sender} · ${timeFormat.format(new Date(message.timestamp))}`));
    const attachments = new Set(message.attachments || []);
    const text = message.text.split('\n').filter((line) => !attachments.has(line.trim())).join('\n').trim();
    if (message.reply_to?.length) row.append(element('div', 'reply-note', '↳ Reply'));
    row.append(element('div', 'bubble', text || (attachments.size ? 'Attachment' : 'Empty message')));
    if (attachments.size && text) row.append(element('div', 'reply-note', `${attachments.size} attachment${attachments.size === 1 ? '' : 's'}`));
    if (message.reactions?.length) row.append(element('div', 'reactions', message.reactions.join(' · ')));
    nodes.push(row);
  }
  container.replaceChildren(...nodes);
  $('earlier').hidden = state.context.start === 0;
  $('later').hidden = state.context.end >= state.context.total;
  $('context-count').textContent = `${state.context.start + 1}–${state.context.end} of ${state.context.total} messages`;
}

function jumpToMatch() {
  const row = $('messages').querySelector('.matched');
  if (row) $('messages-scroll').scrollTop = row.offsetTop - $('messages-scroll').offsetTop - 45;
}

async function openResult(index) {
  const version = ++state.version;
  state.selected = index;
  $('error').hidden = true;
  for (const button of $('results').querySelectorAll('.result')) {
    button.setAttribute('aria-pressed', String(Number(button.dataset.index) === index));
  }
  $('conversation').hidden = true;
  $('conversation-empty').hidden = false;
  $('conversation-empty').querySelector('h2').textContent = 'Opening the conversation…';
  try {
    const data = await api(`/api/search/${state.job}/context/${index}`);
    if (version !== state.version) return;
    state.context = data;
    state.messages = data.messages;
    $('conversation-title').textContent = data.title;
    $('conversation-source').textContent = data.source;
    const summary = state.results.find((hit) => hit.index === index)?.summary;
    $('chunk-summary').hidden = !summary;
    $('chunk-summary-text').textContent = summary || '';
    $('avatar').textContent = data.title.replace(/[^\p{L}\p{N}]/gu, '').slice(0, 2).toUpperCase();
    $('conversation-empty').hidden = true;
    $('conversation').hidden = false;
    document.querySelector('.workspace').classList.add('reading');
    renderMessages();
    jumpToMatch();
  } catch (error) {
    if (version === state.version) showError(error.message);
  } finally {
    if (version === state.version) $('conversation-empty').querySelector('h2').textContent = 'Select a result.';
  }
}

async function loadContext(direction) {
  const version = state.version;
  const current = state.context;
  if (!current) return;
  const earlier = direction === 'earlier';
  const start = earlier ? Math.max(0, current.start - 50) : current.end;
  const end = earlier ? current.start : Math.min(current.total, current.end + 50);
  $('earlier').disabled = true;
  $('later').disabled = true;
  const scroll = $('messages-scroll');
  const anchor = [...$('messages').querySelectorAll('.message')].find((node) => node.getBoundingClientRect().bottom >= scroll.getBoundingClientRect().top);
  const offset = anchor?.getBoundingClientRect().top;
  try {
    const data = await api(`/api/search/${state.job}/context/${state.selected}?start=${start}&end=${end}`);
    if (version !== state.version) return;
    state.messages = earlier ? [...data.messages, ...state.messages] : [...state.messages, ...data.messages];
    state.context = { ...current, start: earlier ? data.start : current.start, end: earlier ? current.end : data.end };
    renderMessages();
    if (anchor) {
      const replacement = [...$('messages').querySelectorAll('.message')].find((node) => node.dataset.messageId === anchor.dataset.messageId);
      if (replacement) scroll.scrollTop += replacement.getBoundingClientRect().top - offset;
    }
  } catch (error) {
    if (version === state.version) showError(error.message);
  } finally {
    $('earlier').disabled = false;
    $('later').disabled = false;
  }
}

$('search-form').addEventListener('submit', (event) => {
  event.preventDefault();
  if (!$('search-button').disabled && $('query').value.trim()) search($('query').value.trim());
});
$('earlier').addEventListener('click', () => loadContext('earlier'));
$('later').addEventListener('click', () => loadContext('later'));
$('retrieval-text').addEventListener('change', () => {
  $('weight-control').hidden = $('retrieval-text').value !== 'both';
});
$('jump').addEventListener('click', jumpToMatch);
$('back').addEventListener('click', () => document.querySelector('.workspace').classList.remove('reading'));
const previousJob = sessionStorage.getItem('message-search-job');
async function initialize() {
  try {
    const settings = await api('/api/config');
    $('retrieval-text').value = settings.retrieval_text;
    $('summary-weight').value = settings.summary_weight;
    $('weight-control').hidden = settings.retrieval_text !== 'both';
    if (previousJob) search('', previousJob);
  } catch (error) {
    showError(error.message);
  }
}
initialize();
