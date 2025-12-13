/*
 * Widget Button Script
 * Sitelere eklenecek standalone script
 * Yuvarlak chat butonu oluşturuyor, tıklanınca iframe açıyor
 */

(function() {
  'use strict';

  // script tag'inden parametreleri al
  const script = document.currentScript || document.querySelector('script[data-agent-id]');
  if (!script) return;

  const agentId = script.getAttribute('data-agent-id');
  const position = script.getAttribute('data-position') || 'bottom-right';
  const baseUrl = script.getAttribute('data-base-url') || window.location.origin;

  if (!agentId) {
    console.error('Widget: agent-id gerekli');
    return;
  }

  // konum stilleri - buton ve widget için
  const positions = {
    'bottom-right': { button: 'bottom: 20px; right: 20px;', widget: 'bottom: 80px; right: 20px;' },
    'bottom-left': { button: 'bottom: 20px; left: 20px;', widget: 'bottom: 80px; left: 20px;' },
    'top-right': { button: 'top: 20px; right: 20px;', widget: 'top: 80px; right: 20px;' },
    'top-left': { button: 'top: 20px; left: 20px;', widget: 'top: 80px; left: 20px;' }
  };

  const pos = positions[position] || positions['bottom-right'];

  // widget state
  let isOpen = false;
  let iframe = null;

  // yuvarlak chat butonu oluştur
  const button = document.createElement('button');
  button.className = 'rag-widget-button';
  button.style.cssText = `
    position: fixed;
    ${pos.button}
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6 0%, #14b8a6 100%);
    border: none;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    cursor: pointer;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s;
  `;
  button.innerHTML = `
    <svg width="28" height="28" fill="none" stroke="white" stroke-width="1.5" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" 
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
    </svg>
  `;

  // hover efekti
  button.addEventListener('mouseenter', () => {
    button.style.transform = 'scale(1.1)';
  });
  button.addEventListener('mouseleave', () => {
    button.style.transform = 'scale(1)';
  });

  // overlay oluştur - widget açıkken arka plan
  const overlay = document.createElement('div');
  overlay.className = 'rag-widget-overlay';
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.2);
    z-index: 9998;
    display: none;
  `;

  // iframe oluştur - chat widget'ı
  function createIframe() {
    if (iframe) return;

    iframe = document.createElement('iframe');
    iframe.src = `${baseUrl}/widget/iframe?agentId=${agentId}`;
    iframe.style.cssText = `
      position: fixed;
      ${pos.widget}
      width: 384px;
      height: 600px;
      border: none;
      border-radius: 8px;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
      z-index: 9999;
      background: white;
    `;
    iframe.setAttribute('allow', 'microphone');

    document.body.appendChild(iframe);
  }

  // widget aç/kapa
  function toggleWidget() {
    isOpen = !isOpen;

    if (isOpen) {
      createIframe();
      overlay.style.display = 'block';
      iframe.style.display = 'block';
    } else {
      overlay.style.display = 'none';
      if (iframe) {
        iframe.style.display = 'none';
      }
    }
  }

  button.addEventListener('click', toggleWidget);
  overlay.addEventListener('click', toggleWidget);

  // DOM'a ekle
  document.body.appendChild(button);
  document.body.appendChild(overlay);

  // iframe'den gelen mesajları dinle (kapatma isteği)
  window.addEventListener('message', (e) => {
    if (e.data && e.data.type === 'close-widget') {
      toggleWidget();
    }
  });
})();
