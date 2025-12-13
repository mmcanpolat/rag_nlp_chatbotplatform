/*
 * Dosya yükleme sayfası
 * PDF, Word, JSON, TXT vs. destekliyor
 */

import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="h-full overflow-y-auto scrollbar-thin">
      <div class="p-6 max-w-4xl mx-auto page-enter">
        <!-- Başlık -->
        <header class="mb-8">
          <h1 class="text-lg font-semibold text-text-primary">Veri Seti Yükle</h1>
          <p class="text-sm text-text-muted">
            Kendi veri setinizi yükleyerek RAG sistemini özelleştirin
          </p>
        </header>

        <!-- Yükleme Alanı -->
        <div class="card p-6 mb-6">
          <h2 class="text-sm font-medium text-text-primary mb-4">Dosya Yükle</h2>
          
          <!-- Dosya Sürükle-Bırak Alanı -->
          <div 
            class="border-2 border-dashed border-border-subtle rounded-xl p-8 text-center
                   hover:border-primary-400 hover:bg-primary-50/30 transition-all cursor-pointer"
            (click)="fileInput.click()"
            (dragover)="onDragOver($event)"
            (drop)="onDrop($event)"
          >
            <input
              #fileInput
              type="file"
              class="hidden"
              accept=".json,.csv,.txt,.md,.pdf,.docx,.doc"
              (change)="onFileSelect($event)"
            />
            
            <div class="w-12 h-12 rounded-xl bg-primary-100 flex items-center justify-center mx-auto mb-4">
              <svg class="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
              </svg>
            </div>
            
            <p class="text-sm text-text-primary mb-1">
              Dosyayı sürükleyip bırakın veya tıklayın
            </p>
            <p class="text-xs text-text-muted">
              Desteklenen formatlar: PDF, DOCX, JSON, TXT, MD (Maks. 50MB)
            </p>
          </div>

          <!-- Seçili Dosya -->
          @if (selectedFile()) {
            <div class="mt-4 p-3 bg-primary-50 rounded-lg flex items-center justify-between">
              <div class="flex items-center gap-3">
                <div class="w-8 h-8 rounded bg-primary-100 flex items-center justify-center">
                  <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                  </svg>
                </div>
                <div>
                  <p class="text-sm font-medium text-text-primary">{{ selectedFile()?.name }}</p>
                  <p class="text-xs text-text-muted">{{ formatFileSize(selectedFile()?.size || 0) }}</p>
                </div>
              </div>
              <button (click)="clearFile()" class="btn-ghost p-1.5 rounded text-text-muted hover:text-status-error">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                </svg>
              </button>
            </div>
          }

          <!-- İndeks Adı -->
          <div class="mt-4">
            <label class="block text-sm font-medium text-text-secondary mb-1.5">
              İndeks Adı
            </label>
            <input
              type="text"
              [(ngModel)]="indexName"
              class="input"
              placeholder="default"
            />
            <p class="text-xs text-text-muted mt-1">
              Farklı veri setlerini ayırmak için benzersiz bir isim verin
            </p>
          </div>

          <!-- Yükle Butonu -->
          <div class="mt-6">
            <button
              (click)="uploadFile()"
              [disabled]="!selectedFile() || uploading()"
              class="btn btn-primary w-full justify-center"
              [class.opacity-50]="!selectedFile() || uploading()"
            >
              @if (uploading()) {
                <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                <span>Yükleniyor...</span>
              } @else {
                <span>Veri Setini Yükle ve İşle</span>
              }
            </button>
          </div>
        </div>

        <!-- Başarı/Hata Mesajı -->
        @if (message()) {
          <div class="mb-6 p-4 rounded-lg" 
               [class]="success() ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'">
            {{ message() }}
          </div>
        }

        <!-- JSON Yapıştırma -->
        <div class="card p-6 mb-6">
          <h2 class="text-sm font-medium text-text-primary mb-4">JSON Yapıştır</h2>
          
          <div class="mb-4">
            <p class="text-xs text-text-muted mb-2">Beklenen format:</p>
            <pre class="bg-primary-50 p-3 rounded-lg text-xs font-mono text-text-secondary overflow-x-auto">
[
  {{ '{' }}
    "question": "Soru metni",
    "answer": "Cevap metni",
    "context": "Bağlam bilgisi (opsiyonel)"
  {{ '}' }}
]</pre>
          </div>
          
          <textarea
            [(ngModel)]="jsonData"
            class="input font-mono text-xs"
            rows="8"
            placeholder='[{"question": "Örnek soru?", "answer": "Örnek cevap", "context": "Bağlam"}]'
          ></textarea>
          
          <div class="mt-4 flex gap-3">
            <input
              type="text"
              [(ngModel)]="jsonIndexName"
              class="input flex-1"
              placeholder="İndeks adı (default)"
            />
            <button
              (click)="uploadJson()"
              [disabled]="!jsonData.trim() || uploading()"
              class="btn btn-secondary"
            >
              JSON Yükle
            </button>
          </div>
        </div>

        <!-- Mevcut İndeksler -->
        <div class="card p-6">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-sm font-medium text-text-primary">Mevcut İndeksler</h2>
            <button (click)="loadIndices()" class="btn-ghost text-xs px-2 py-1 rounded">
              Yenile
            </button>
          </div>
          
          @if (api.indices().length > 0) {
            <div class="space-y-2">
              @for (index of api.indices(); track index.name) {
                <div class="flex items-center justify-between p-3 bg-primary-50 rounded-lg">
                  <div class="flex items-center gap-3">
                    <div class="w-8 h-8 rounded bg-primary-100 flex items-center justify-center">
                      <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"/>
                      </svg>
                    </div>
                    <div>
                      <p class="text-sm font-medium text-text-primary">{{ index.name }}</p>
                      <p class="text-xs text-text-muted">{{ index.documentCount }} döküman</p>
                    </div>
                  </div>
                  <button
                    (click)="api.setActiveIndex(index.name)"
                    class="text-xs font-medium"
                    [class]="api.activeIndex() === index.name ? 'text-primary-600' : 'text-text-muted hover:text-primary-600'"
                  >
                    {{ api.activeIndex() === index.name ? 'Aktif' : 'Seç' }}
                  </button>
                </div>
              }
            </div>
          } @else {
            <p class="text-sm text-text-muted text-center py-4">
              Henüz indeks oluşturulmamış. Veri seti yükleyerek başlayın.
            </p>
          }
        </div>
      </div>
    </div>
  `
})
export class UploadComponent {
  // Durum sinyalleri
  selectedFile = signal<File | null>(null);
  uploading = signal(false);
  message = signal<string | null>(null);
  success = signal(false);
  
  // Form alanları
  indexName = 'default';
  jsonData = '';
  jsonIndexName = 'default';

  constructor(public api: ApiService) {
    // İndeksleri yükle
    this.loadIndices();
  }

  /** İndeks listesini yükler */
  loadIndices(): void {
    this.api.loadIndices().subscribe();
  }

  /** Dosya seçimi işleyicisi */
  onFileSelect(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile.set(input.files[0]);
      this.message.set(null);
    }
  }

  /** Sürükle-bırak işleyicisi */
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  /** Dosya bırakma işleyicisi */
  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    
    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      const file = event.dataTransfer.files[0];
      const ext = file.name.split('.').pop()?.toLowerCase();
      
      if (['json', 'csv', 'txt', 'md', 'pdf', 'docx', 'doc'].includes(ext || '')) {
        this.selectedFile.set(file);
        this.message.set(null);
      } else {
        this.message.set('Desteklenmeyen dosya formatı. PDF, DOCX, JSON, TXT veya MD yükleyebilirsiniz.');
        this.success.set(false);
      }
    }
  }

  /** Dosya seçimini temizler */
  clearFile(): void {
    this.selectedFile.set(null);
    this.message.set(null);
  }

  /** Dosyayı yükler */
  uploadFile(): void {
    const file = this.selectedFile();
    if (!file) return;

    this.uploading.set(true);
    this.message.set(null);

    this.api.uploadFile(file, this.indexName || 'default').subscribe({
      next: (response) => {
        this.uploading.set(false);
        this.success.set(true);
        this.message.set(`Başarılı! ${response.data?.document_count || 'Veri seti'} döküman işlendi.`);
        this.clearFile();
        this.loadIndices();
      },
      error: (err) => {
        this.uploading.set(false);
        this.success.set(false);
        this.message.set(err.message);
      }
    });
  }

  /** JSON verisini yükler */
  uploadJson(): void {
    if (!this.jsonData.trim()) return;

    try {
      const data = JSON.parse(this.jsonData);
      
      if (!Array.isArray(data)) {
        this.message.set('JSON bir array olmalıdır.');
        this.success.set(false);
        return;
      }

      this.uploading.set(true);
      this.message.set(null);

      this.api.uploadJsonData(data, this.jsonIndexName || 'default').subscribe({
        next: (response) => {
          this.uploading.set(false);
          this.success.set(true);
          this.message.set(`Başarılı! ${data.length} döküman işlendi.`);
          this.jsonData = '';
          this.loadIndices();
        },
        error: (err) => {
          this.uploading.set(false);
          this.success.set(false);
          this.message.set(err.message);
        }
      });

    } catch (e) {
      this.message.set('Geçersiz JSON formatı. Lütfen kontrol edin.');
      this.success.set(false);
    }
  }

  /** Dosya boyutunu formatlar */
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
}

