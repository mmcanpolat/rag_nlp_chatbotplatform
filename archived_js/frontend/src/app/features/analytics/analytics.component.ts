/*
 * Analiz sayfası - model karşılaştırma grafikleri
 */

import { Component, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService, EvaluationResult } from '../../core/services/api.service';

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="h-full overflow-y-auto scrollbar-thin">
      <div class="p-6 max-w-6xl mx-auto page-enter">
        <!-- Başlık -->
        <header class="mb-6">
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-lg font-semibold text-text-primary">Model Analizi</h1>
              <p class="text-sm text-text-muted">Performans metrikleri ve karşılaştırmalar</p>
            </div>
            
            <button
              (click)="runBenchmark()"
              [disabled]="loading()"
              class="btn btn-primary"
            >
              @if (loading()) {
                <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                <span>Değerlendiriliyor...</span>
              } @else {
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                </svg>
                <span>Değerlendirme Başlat</span>
              }
            </button>
          </div>
        </header>

        <!-- Metrik Özeti -->
        @if (api.evaluationResults()) {
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <!-- Cosine Similarity -->
            <div class="card p-4">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs text-text-muted">Ortalama Cosine Sim.</span>
                <div class="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center">
                  <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2z"/>
                  </svg>
                </div>
              </div>
              <p class="text-2xl font-bold text-text-primary">
                {{ getAverageMetric('avg_cosine_similarity').toFixed(3) }}
              </p>
              <p class="text-xs text-text-muted mt-1">Semantik benzerlik</p>
            </div>

            <!-- ROUGE Score -->
            <div class="card p-4">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs text-text-muted">Ortalama ROUGE</span>
                <div class="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center">
                  <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/>
                  </svg>
                </div>
              </div>
              <p class="text-2xl font-bold text-text-primary">
                {{ getAverageMetric('avg_rouge_score').toFixed(3) }}
              </p>
              <p class="text-xs text-text-muted mt-1">N-gram örtüşmesi</p>
            </div>

            <!-- F1 Score -->
            <div class="card p-4">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs text-text-muted">Ortalama F1</span>
                <div class="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center">
                  <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                </div>
              </div>
              <p class="text-2xl font-bold text-text-primary">
                {{ getAverageMetric('avg_f1_score').toFixed(3) }}
              </p>
              <p class="text-xs text-text-muted mt-1">Precision-Recall dengesi</p>
            </div>

            <!-- Accuracy -->
            <div class="card p-4">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs text-text-muted">Ortalama Doğruluk</span>
                <div class="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center">
                  <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M5 13l4 4L19 7"/>
                  </svg>
                </div>
              </div>
              <p class="text-2xl font-bold text-text-primary">
                {{ (getAverageMetric('accuracy') * 100).toFixed(1) }}%
              </p>
              <p class="text-xs text-text-muted mt-1">Eşik tabanlı sınıflandırma</p>
            </div>
          </div>

          <!-- Model Bazlı Detaylar -->
          <div class="card p-5 mb-6">
            <h2 class="text-sm font-medium text-text-primary mb-4">Model Karşılaştırması</h2>
            
            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="text-left text-text-muted border-b border-border-subtle">
                    <th class="pb-3 font-medium">Model</th>
                    <th class="pb-3 font-medium text-center">Cosine</th>
                    <th class="pb-3 font-medium text-center">ROUGE</th>
                    <th class="pb-3 font-medium text-center">F1</th>
                    <th class="pb-3 font-medium text-center">Accuracy</th>
                    <th class="pb-3 font-medium text-center">Yanıt Süresi</th>
                  </tr>
                </thead>
                <tbody>
                  @for (model of getModels(); track model) {
                    <tr class="border-b border-border-subtle last:border-0">
                      <td class="py-3 font-medium text-text-primary">{{ model }}</td>
                      <td class="py-3 text-center">
                        <span class="inline-flex items-center px-2 py-0.5 rounded bg-primary-50 text-primary-600 text-xs font-medium">
                          {{ getMetric(model, 'avg_cosine_similarity').toFixed(3) }}
                        </span>
                      </td>
                      <td class="py-3 text-center">
                        {{ getMetric(model, 'avg_rouge_score').toFixed(3) }}
                      </td>
                      <td class="py-3 text-center">
                        {{ getMetric(model, 'avg_f1_score').toFixed(3) }}
                      </td>
                      <td class="py-3 text-center">
                        {{ (getMetric(model, 'accuracy') * 100).toFixed(1) }}%
                      </td>
                      <td class="py-3 text-center text-text-muted">
                        {{ getMetric(model, 'avg_response_time_ms').toFixed(0) }}ms
                      </td>
                    </tr>
                  }
                </tbody>
              </table>
            </div>
          </div>
        }

        <!-- Grafikler -->
        <div class="mb-6">
          <h2 class="text-sm font-medium text-text-primary mb-4">Görselleştirmeler</h2>
          
          @if (plots().length > 0) {
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
              @for (plot of plots(); track plot.name) {
                <div class="card p-4">
                  <h3 class="text-sm font-medium text-text-secondary mb-3">
                    {{ getPlotTitle(plot.name) }}
                  </h3>
                  <img 
                    [src]="plot.path" 
                    [alt]="plot.name"
                    class="w-full rounded-lg"
                    loading="lazy"
                  />
                </div>
              }
            </div>
          } @else {
            <div class="card p-8 text-center">
              <div class="w-12 h-12 rounded-xl bg-primary-100 flex items-center justify-center mx-auto mb-4">
                <svg class="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                </svg>
              </div>
              <p class="text-sm text-text-muted">
                Henüz grafik yok. Değerlendirme başlatarak grafikleri oluşturabilirsiniz.
              </p>
            </div>
          }
        </div>

        <!-- Hata mesajı -->
        @if (error()) {
          <div class="bg-red-50 border border-red-200 rounded-lg p-4">
            <p class="text-sm text-red-600">{{ error() }}</p>
          </div>
        }
      </div>
    </div>
  `
})
export class AnalyticsComponent implements OnInit {
  loading = signal(false);
  error = signal<string | null>(null);
  plots = signal<{ name: string; path: string }[]>([]);

  constructor(public api: ApiService) {}

  ngOnInit(): void {
    this.loadData();
  }

  /** Verileri yükler */
  loadData(): void {
    // Mevcut sonuçları yükle
    this.api.getEvaluationResults().subscribe({
      error: () => {} // Sonuç yoksa hata gösterme
    });

    // Grafikleri yükle
    this.api.getPlots().subscribe({
      next: (response) => {
        this.plots.set(response.plots);
      },
      error: () => {}
    });
  }

  /** Benchmark çalıştırır */
  runBenchmark(): void {
    this.loading.set(true);
    this.error.set(null);

    this.api.runBenchmark().subscribe({
      next: () => {
        this.loading.set(false);
        // Grafikleri yeniden yükle
        setTimeout(() => this.loadData(), 1000);
      },
      error: (err) => {
        this.loading.set(false);
        this.error.set(err.message);
      }
    });
  }

  /** Model listesini döndürür */
  getModels(): string[] {
    const results = this.api.evaluationResults();
    return results ? Object.keys(results) : [];
  }

  /** Belirli bir metriği döndürür */
  getMetric(model: string, metric: string): number {
    const results = this.api.evaluationResults();
    if (!results || !results[model]) return 0;
    return (results[model] as any)[metric] || 0;
  }

  /** Ortalama metrik hesaplar */
  getAverageMetric(metric: string): number {
    const results = this.api.evaluationResults();
    if (!results) return 0;

    const values = Object.values(results)
      .map((r: any) => r[metric] || 0)
      .filter(v => v > 0);

    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  /** Grafik başlığını döndürür */
  getPlotTitle(filename: string): string {
    const titles: Record<string, string> = {
      'confusion_matrix.png': 'Karışıklık Matrisi',
      'metrics_comparison.png': 'Metrik Karşılaştırması',
      'response_time.png': 'Yanıt Süreleri',
      'radar_chart.png': 'Radar Grafiği'
    };
    return titles[filename] || filename;
  }
}
