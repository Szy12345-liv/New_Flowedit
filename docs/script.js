/* Static, dependency-free project page. All scientific images are existing results. */
(() => {
  'use strict';
  const data = window.DECFLOWEDIT_EXAMPLES;
  const $ = (selector) => document.querySelector(selector);
  const lightbox = $('#lightbox');
  let lastImageButton;

  $('.nav-toggle').addEventListener('click', (event) => {
    const expanded = event.currentTarget.getAttribute('aria-expanded') === 'true';
    event.currentTarget.setAttribute('aria-expanded', String(!expanded));
    $('#nav-links').classList.toggle('is-open', !expanded);
  });
  $('#nav-links').addEventListener('click', (event) => {
    if (event.target.closest('a')) {
      $('#nav-links').classList.remove('is-open');
      $('.nav-toggle').setAttribute('aria-expanded', 'false');
    }
  });

  document.addEventListener('click', (event) => {
    const button = event.target.closest('[data-full]');
    if (!button) return;
    lastImageButton = button;
    const thumbnail = button.querySelector('img');
    $('#lightbox-image').src = button.dataset.full;
    $('#lightbox-image').alt = thumbnail.alt;
    $('#lightbox-caption').textContent = thumbnail.alt;
    lightbox.showModal();
    document.body.style.overflow = 'hidden';
  });
  $('#close-lightbox').addEventListener('click', () => lightbox.close());
  lightbox.addEventListener('click', (event) => {
    if (event.target !== lightbox) return;
    const rect = lightbox.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) lightbox.close();
  });
  lightbox.addEventListener('close', () => {
    document.body.style.overflow = '';
    lastImageButton?.focus({preventScroll: true});
  });

  $('#copy-citation').addEventListener('click', async () => {
    const text = $('#citation-text').textContent.trim();
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const field = document.createElement('textarea');
        field.value = text;
        field.style.position = 'fixed';
        field.style.opacity = '0';
        document.body.append(field);
        field.select();
        const copied = document.execCommand('copy');
        field.remove();
        if (!copied) throw new Error('Clipboard unavailable');
      }
      $('#copy-status').textContent = 'Citation copied.';
    } catch {
      $('#copy-status').textContent = 'Select the citation text above to copy it.';
    }
  });

  if (!data?.examples?.length) {
    $('#comparison-note').textContent = 'The complete comparison is available in the figure below.';
    $('.full-comparison').open = true;
    document.querySelectorAll('[data-example], [data-backbone], #comparison-example, #previous-example, #next-example').forEach((button) => { button.disabled = true; });
    return;
  }

  const exampleById = new Map(data.examples.map((example) => [example.id, example]));
  let comparisonId = data.examples[0].id;
  let backboneId = 'flux';

  function figure(src, label, example, backbone, ours = false) {
    const item = document.createElement('figure');
    if (ours) item.className = 'ours';
    const caption = document.createElement('figcaption');
    caption.textContent = label;
    if (ours) {
      const note = document.createElement('span');
      note.className = 'method-note';
      note.textContent = ' (ours)';
      caption.append(note);
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'image-button';
    button.dataset.full = src;
    button.setAttribute('aria-label', `Enlarge ${label}: ${example.title}`);
    const img = document.createElement('img');
    img.src = src;
    img.width = 512;
    img.height = 512;
    img.alt = label === 'Source' ? `Source: ${example.source_prompt}` : `${label}, ${backbone}: ${example.target_prompt}`;
    button.append(img);
    item.append(caption, button);
    return item;
  }

  function promptPair(container, example) {
    const sourceWords = example.source_prompt.split(/\s+/);
    const targetWords = example.target_prompt.split(/\s+/);
    let prefix = 0;
    while (prefix < Math.min(sourceWords.length, targetWords.length) && sourceWords[prefix] === targetWords[prefix]) prefix++;
    let suffix = 0;
    while (suffix < Math.min(sourceWords.length, targetWords.length) - prefix && sourceWords[sourceWords.length - 1 - suffix] === targetWords[targetWords.length - 1 - suffix]) suffix++;
    function line(words, labelText) {
      const p = document.createElement('p');
      const label = document.createElement('span');
      label.className = 'prompt-label';
      label.textContent = labelText;
      p.append(label);
      words.forEach((word, index) => {
        if (index) p.append(' ');
        if (index >= prefix && index < words.length - suffix) {
          const mark = document.createElement('mark');
          mark.textContent = word;
          p.append(mark);
        } else p.append(word);
      });
      return p;
    }
    container.replaceChildren(line(sourceWords, 'Source prompt'), line(targetWords, 'Target prompt'));
  }

  function renderHero(id) {
    const example = exampleById.get(id);
    if (!example) return;
    $('#hero-grid').replaceChildren(
      figure(example.source, 'Source', example, 'FLUX.1-dev'),
      figure(example.images.flux.flowedit, 'FlowEdit', example, 'FLUX.1-dev'),
      figure(example.images.flux.decflowedit, 'DecFlowEdit', example, 'FLUX.1-dev', true)
    );
    promptPair($('#hero-prompt'), example);
    document.querySelectorAll('[data-example]').forEach((button) => {
      const selected = button.dataset.example === id;
      button.classList.toggle('is-active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
  }

  function renderComparison() {
    const example = exampleById.get(comparisonId);
    const backbone = data.backbones.find((item) => item.id === backboneId);
    const grid = $('#comparison-grid');
    grid.style.setProperty('--columns', backbone.methods.length + 1);
    grid.replaceChildren(figure(example.source, 'Source', example, backbone.label), ...backbone.methods.map((method) => figure(example.images[backboneId][method.id], method.label, example, backbone.label, method.id === 'decflowedit')));
    promptPair($('#comparison-prompt'), example);
    $('#comparison-example').value = comparisonId;
    $('#example-counter').textContent = `${data.examples.findIndex((item) => item.id === comparisonId) + 1} / ${data.examples.length}`;
    document.querySelectorAll('[data-backbone]').forEach((button) => {
      const selected = button.dataset.backbone === backboneId;
      button.classList.toggle('is-active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    $('.comparison-scroll').scrollLeft = 0;
  }

  $('#hero-picker').addEventListener('click', (event) => {
    const button = event.target.closest('[data-example]');
    if (button) renderHero(button.dataset.example);
  });
  $('#backbone-picker').addEventListener('click', (event) => {
    const button = event.target.closest('[data-backbone]');
    if (button) { backboneId = button.dataset.backbone; renderComparison(); }
  });
  $('#comparison-example').addEventListener('change', (event) => { comparisonId = event.target.value; renderComparison(); });
  function advance(delta) {
    const current = data.examples.findIndex((example) => example.id === comparisonId);
    comparisonId = data.examples[(current + delta + data.examples.length) % data.examples.length].id;
    renderComparison();
  }
  $('#previous-example').addEventListener('click', () => advance(-1));
  $('#next-example').addEventListener('click', () => advance(1));
  renderHero('cake');
  renderComparison();
})();
