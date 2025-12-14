/*
 * Agent oluşturma - veri yükle, embedding seç
 */

import { Component, signal, OnInit, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService, Agent, EmbeddingModel } from '../../core/services/api.service';

@Component({
  selector: 'app-agents',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="h-full overflow-y-auto scrollbar-thin">
      <div class="p-6 max-w-4xl mx-auto page-enter">
        <!-- Başlık -->
        <header class="mb-6">
          <h1 class="text-lg font-semibold text-text-primary">Agent Oluştur</h1>
          <p class="text-sm text-text-muted">Veri yükleyerek özel chatbot oluşturun</p>
        </header>

        <!-- Agent Oluşturma Formu -->
        <div class="card p-6 mb-6">
          <h2 class="text-sm font-medium text-text-primary mb-4">Yeni Agent</h2>
          
          <div class="space-y-4">
            <!-- Agent Adı -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Agent Adı
              </label>
              <input
                type="text"
                [(ngModel)]="agentName"
                class="input"
                placeholder="Örn: Müşteri Destek Botu"
              />
            </div>

            <!-- Embedding Model Seçimi -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Embedding Modeli
              </label>
              <div class="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  (click)="embeddingModel = 'text-embedding-3-large'"
                  class="p-3 rounded-lg border text-left transition-all"
                  [class]="embeddingModel === 'text-embedding-3-large' 
                    ? 'border-primary-500 bg-primary-50' 
                    : 'border-border-subtle hover:border-primary-300'"
                >
                  <p class="text-sm font-medium text-text-primary">OpenAI</p>
                  <p class="text-xs text-text-muted">text-embedding-3-large</p>
                </button>
                
                <button
                  type="button"
                  (click)="embeddingModel = 'paraphrase-multilingual-MiniLM-L12-v2'"
                  class="p-3 rounded-lg border text-left transition-all"
                  [class]="embeddingModel === 'paraphrase-multilingual-MiniLM-L12-v2' 
                    ? 'border-primary-500 bg-primary-50' 
                    : 'border-border-subtle hover:border-primary-300'"
                >
                  <p class="text-sm font-medium text-text-primary">HuggingFace</p>
                  <p class="text-xs text-text-muted">paraphrase-multilingual</p>
                </button>
              </div>
            </div>

            <!-- Veri Kaynağı Seçimi -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Veri Kaynağı
              </label>
              <div class="flex gap-2 mb-3">
                <button
                  type="button"
                  (click)="sourceType = 'file'"
                  class="px-3 py-1.5 text-sm rounded-md transition-all"
                  [class]="sourceType === 'file' 
                    ? 'bg-primary-600 text-white' 
                    : 'bg-primary-50 text-text-secondary hover:bg-primary-100'"
                >
                  Dosya Yükle
                </button>
                <button
                  type="button"
                  (click)="sourceType = 'url'"
                  class="px-3 py-1.5 text-sm rounded-md transition-all"
                  [class]="sourceType === 'url' 
                    ? 'bg-primary-600 text-white' 
                    : 'bg-primary-50 text-text-secondary hover:bg-primary-100'"
                >
                  Web Sitesi
                </button>
              </div>
              
              <!-- Dosya yükleme -->
              @if (sourceType === 'file') {
                <div 
                  class="border-2 border-dashed border-border-subtle rounded-lg p-6 text-center cursor-pointer
                         hover:border-primary-400 hover:bg-primary-50/30 transition-all"
                  (click)="fileInput.click()"
                  (dragover)="$event.preventDefault()"
                  (drop)="onDrop($event)"
                >
                  <input
                    #fileInput
                    type="file"
                    class="hidden"
                    accept=".pdf,.docx,.doc,.txt,.md,.json,.csv"
                    (change)="onFileSelect($event)"
                  />
                  
                  @if (selectedFile()) {
                    <div class="flex items-center justify-center gap-3">
                      <svg class="w-8 h-8 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                      </svg>
                      <div class="text-left">
                        <p class="text-sm font-medium text-text-primary">{{ selectedFile()?.name }}</p>
                        <p class="text-xs text-text-muted">{{ formatSize(selectedFile()?.size || 0) }}</p>
                      </div>
                      <button (click)="clearFile($event)" class="p-1 hover:bg-primary-100 rounded">
                        <svg class="w-4 h-4 text-text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                        </svg>
                      </button>
                    </div>
                  } @else {
                    <svg class="w-10 h-10 text-primary-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
                    </svg>
                    <p class="text-sm text-text-primary">Dosya sürükle veya tıkla</p>
                    <p class="text-xs text-text-muted">PDF, DOCX, TXT, JSON, CSV</p>
                  }
                </div>
              }
              
              <!-- URL girişi -->
              @if (sourceType === 'url') {
                <input
                  type="url"
                  [(ngModel)]="websiteUrl"
                  class="input"
                  placeholder="https://example.com/sayfa"
                />
              }
            </div>

            <!-- Oluştur Butonu -->
            <button
              (click)="createAgent()"
              [disabled]="!canCreate() || creating()"
              class="btn btn-primary w-full justify-center"
              [class.opacity-50]="!canCreate() || creating()"
            >
              @if (creating()) {
                <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                <span>İşleniyor...</span>
              } @else {
                <span>Agent Oluştur</span>
              }
            </button>
          </div>
        </div>

        <!-- Mesaj -->
        @if (message()) {
          <div class="mb-6 p-4 rounded-lg" 
               [class]="success() ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'">
            {{ message() }}
          </div>
        }

        <!-- Mevcut Agentlar -->
        <div class="card p-6">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-sm font-medium text-text-primary">Agentlarınız</h2>
            <button (click)="loadAgents()" class="text-xs text-text-muted hover:text-primary-600">
              Yenile
            </button>
          </div>
          
          @if (api.agents().length > 0) {
            <div class="space-y-3">
              @for (agent of api.agents(); track agent.id) {
                <div class="flex items-center justify-between p-3 bg-primary-50 rounded-lg">
                  <div class="flex items-center gap-3">
                    <div class="w-8 h-8 rounded bg-primary-100 flex items-center justify-center">
                      <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                      </svg>
                    </div>
                    <div>
                      <p class="text-sm font-medium text-text-primary">{{ agent.name }}</p>
                      <p class="text-xs text-text-muted">{{ agent.embeddingModel || 'default' }}</p>
                    </div>
                  </div>
                  <div class="flex items-center gap-2">
                    <button (click)="openWidgetEmbed(agent.id)" 
                            class="text-xs text-primary-600 hover:underline px-2 py-1 rounded hover:bg-primary-50">
                      Widget Kodu
                    </button>
                    <button (click)="useAgent(agent)" class="text-xs text-primary-600 hover:underline">
                      Kullan
                    </button>
                    <button (click)="deleteAgent(agent.id)" class="p-1.5 text-text-muted hover:text-red-500">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                      </svg>
                    </button>
                  </div>
                </div>
              }
            </div>
          } @else {
            <p class="text-sm text-text-muted text-center py-4">
              Henüz agent yok. Yukarıdan oluşturun.
            </p>
          }
        </div>
      </div>
    </div>
  `
})
export class AgentsComponent implements OnInit {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;
  
  agentName = '';
  embeddingModel: EmbeddingModel = 'paraphrase-multilingual-MiniLM-L12-v2';
  sourceType: 'file' | 'url' = 'file';
  websiteUrl = '';
  
  selectedFile = signal<File | null>(null);
  creating = signal(false);
  message = signal<string | null>(null);
  success = signal(false);

  constructor(public api: ApiService, private router: Router) {}

  ngOnInit(): void {
    this.loadAgents();
    this.api.loadIndices().subscribe();
  }

  loadAgents(): void {
    this.api.loadAgents().subscribe();
  }

  onFileSelect(e: Event): void {
    const input = e.target as HTMLInputElement;
    if (input.files?.[0]) {
      this.selectedFile.set(input.files[0]);
      this.message.set(null);
    }
  }

  onDrop(e: DragEvent): void {
    e.preventDefault();
    if (e.dataTransfer?.files?.[0]) {
      this.selectedFile.set(e.dataTransfer.files[0]);
      this.message.set(null);
    }
  }

  clearFile(e: Event): void {
    e.stopPropagation();
    this.selectedFile.set(null);
    if (this.fileInput) {
      this.fileInput.nativeElement.value = '';
    }
  }

  canCreate(): boolean {
    if (!this.agentName.trim()) return false;
    if (this.sourceType === 'file' && !this.selectedFile()) return false;
    if (this.sourceType === 'url' && !this.websiteUrl.trim()) return false;
    return true;
  }

  async createAgent(): Promise<void> {
    if (!this.canCreate()) return;

    this.creating.set(true);
    this.message.set(null);

    const indexName = `agent_${Date.now()}`;
    let dataSource: string | null = null;

    try {
      // dosya yükleme - önce dosyayı backend'e yükle
      if (this.sourceType === 'file') {
        const file = this.selectedFile()!;
        const uploadRes = await this.api.uploadFile(file, indexName, this.embeddingModel).toPromise();
        if (!uploadRes?.success) {
          throw new Error(uploadRes?.error || 'Dosya yükleme başarısız');
        }
        // backend'den dönen dosya adını al
        dataSource = uploadRes?.data?.filePath || file.name;
      } else {
        // URL için direkt URL'i gönder
        dataSource = this.websiteUrl.trim();
      }

      // agent oluştur - backend veri kaynağını işleyecek
      const agentRes = await this.api.createAgent({
        name: this.agentName,
        indexName,
        embeddingModel: this.embeddingModel,
        dataSourceType: this.sourceType,
        dataSource: dataSource
      }).toPromise();

      if (!agentRes?.success) {
        throw new Error(agentRes?.error || 'Agent oluşturma başarısız');
      }

      this.success.set(true);
      this.message.set('Agent başarıyla oluşturuldu ve veri işlendi!');
      this.resetForm();
      this.loadAgents();

    } catch (err: any) {
      this.success.set(false);
      this.message.set(err?.message || err?.error || 'Hata oluştu');
    }

    this.creating.set(false);
  }

  useAgent(agent: Agent): void {
    this.api.setActiveIndex(agent.indexName);
    window.location.href = '/chat';
  }

  openWidgetEmbed(agentId: string): void {
    this.router.navigate(['/widget/embed', agentId]);
  }

  deleteAgent(id: string): void {
    if (!confirm('Bu agenti silmek istediğinizden emin misiniz?')) return;
    this.api.deleteAgent(id).subscribe();
  }

  resetForm(): void {
    this.agentName = '';
    this.selectedFile.set(null);
    this.websiteUrl = '';
    if (this.fileInput) {
      this.fileInput.nativeElement.value = '';
    }
  }

  formatSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
  }
}
