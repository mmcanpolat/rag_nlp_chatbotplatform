// widget embed kodu sayfası - agent için iframe kodu oluşturuyor

import { Component, signal, computed, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-widget-embed',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="h-full overflow-y-auto scrollbar-thin">
      <div class="p-6 max-w-4xl mx-auto page-enter">
        <header class="mb-6">
          <h1 class="text-lg font-semibold text-text-primary">Widget Embed Kodu</h1>
          <p class="text-sm text-text-muted">Agent'ınızı sitenize ekleyin</p>
        </header>

        @if (agent()) {
          <div class="card p-6 space-y-6">
            <!-- konum seçimi -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-3">
                Widget Konumu
              </label>
              <div class="grid grid-cols-4 gap-3">
                @for (pos of positions; track pos.value) {
                  <button
                    type="button"
                    (click)="selectedPosition.set(pos.value)"
                    class="p-4 rounded-lg border text-center transition-all"
                    [class]="selectedPosition() === pos.value 
                      ? 'border-primary-500 bg-primary-50' 
                      : 'border-border-subtle hover:border-primary-300'"
                  >
                    <div class="text-2xl mb-2">{{ pos.icon }}</div>
                    <p class="text-xs font-medium text-text-primary">{{ pos.label }}</p>
                  </button>
                }
              </div>
            </div>

            <!-- embed kodu -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-2">
                Embed Kodu
              </label>
              <div class="relative">
                <textarea
                  readonly
                  [value]="embedCode()"
                  class="input font-mono text-xs h-32 resize-none"
                ></textarea>
                <button
                  (click)="copyCode()"
                  class="absolute top-2 right-2 btn btn-secondary text-xs px-3 py-1.5"
                >
                  Kopyala
                </button>
              </div>
              <p class="text-xs text-text-muted mt-2">
                Bu kodu sitenizin &lt;/body&gt; etiketinden önce ekleyin
              </p>
            </div>

            <!-- önizleme -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-2">
                Önizleme
              </label>
              <div class="border border-border-subtle rounded-lg p-8 bg-gray-50 min-h-[400px] relative">
                <div class="text-sm text-text-muted mb-4">Sitenizin görünümü</div>
                <div class="absolute" [style]="getPositionStyle()">
                  <div class="w-14 h-14 rounded-full bg-gradient-to-br from-blue-500 to-teal-500 
                              shadow-lg flex items-center justify-center cursor-pointer hover:scale-110 transition-transform">
                    <svg class="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                    </svg>
                  </div>
                </div>
              </div>
            </div>
          </div>
        } @else {
          <div class="card p-8 text-center">
            <p class="text-sm text-text-muted">Agent bulunamadı</p>
          </div>
        }
      </div>
    </div>
  `
})
export class WidgetEmbedComponent implements OnInit {
  agent = signal<any>(null);
  selectedPosition = signal<string>('bottom-right');
  
  // widget konumları
  positions = [
    { value: 'bottom-right', label: 'Sağ Alt', icon: '↘️' },
    { value: 'bottom-left', label: 'Sol Alt', icon: '↙️' },
    { value: 'top-right', label: 'Sağ Üst', icon: '↗️' },
    { value: 'top-left', label: 'Sol Üst', icon: '↖️' }
  ];

  constructor(
    private route: ActivatedRoute,
    private api: ApiService
  ) {}

  ngOnInit(): void {
    const agentId = this.route.snapshot.params['id'];
    if (agentId) {
      // agentları yükle ve bul
      this.api.loadAgents().subscribe(() => {
        const agent = this.api.agents().find(a => a.id === agentId);
        if (agent) {
          this.agent.set(agent);
        }
      });
    }
  }

  // embed kodunu oluştur - script tag'i olarak döndürüyor
  embedCode = computed(() => {
    const agent = this.agent();
    if (!agent) return '';
    
    const pos = this.selectedPosition();
    const baseUrl = window.location.origin;
    
    // script tag'i oluştur, widget-button.js'i yükle
    return `<script>
  (function() {
    var script = document.createElement('script');
    script.src = '${baseUrl}/widget-button.js';
    script.setAttribute('data-agent-id', '${agent.id}');
    script.setAttribute('data-position', '${pos}');
    script.setAttribute('data-base-url', '${baseUrl}');
    script.async = true;
    document.body.appendChild(script);
  })();
</script>`;
  });

  // seçilen konuma göre stil döndür
  getPositionStyle(): string {
    const pos = this.selectedPosition();
    const styles: Record<string, string> = {
      'bottom-right': 'bottom: 20px; right: 20px;',
      'bottom-left': 'bottom: 20px; left: 20px;',
      'top-right': 'top: 20px; right: 20px;',
      'top-left': 'top: 20px; left: 20px;'
    };
    return styles[pos] || styles['bottom-right'];
  }

  // embed kodunu kopyala
  copyCode(): void {
    navigator.clipboard.writeText(this.embedCode());
    alert('Kod kopyalandı!');
  }
}
