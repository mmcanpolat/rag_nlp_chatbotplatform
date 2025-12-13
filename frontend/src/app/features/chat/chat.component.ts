// sohbet ekranı - agent seçimi ve model seçimi ile chat yapıyor

import { Component, signal, ViewChild, ElementRef, AfterViewChecked, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, ModelType, Agent } from '../../core/services/api.service';
import { ContextCardComponent } from '../../shared/components/context-card.component';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, ContextCardComponent],
  template: `
    <div class="h-full flex flex-col overflow-hidden">
      <!-- Üst Bar -->
      <header class="bg-white border-b border-border-subtle p-4">
        <div class="flex items-center justify-between">
          <div class="flex-1">
            <h1 class="text-lg font-semibold text-text-primary">Sohbet</h1>
            <div class="flex items-center gap-4 mt-1">
              <!-- Agent Seçici -->
              <div class="flex items-center gap-2">
                <label class="text-xs text-text-muted">Agent:</label>
                <select
                  [(ngModel)]="selectedAgentId"
                  (change)="onAgentChange()"
                  class="text-xs px-2 py-1 border border-border-subtle rounded-md bg-white"
                >
                  <option value="">Varsayılan</option>
                  @for (agent of agents(); track agent.id) {
                    <option [value]="agent.id">{{ agent.name }}</option>
                  }
                </select>
              </div>
              <p class="text-xs text-text-muted">
                Index: <strong>{{ currentIndexName() }}</strong>
              </p>
            </div>
          </div>
          
          <!-- Model Seçici -->
          <div class="flex items-center gap-1 bg-primary-50 p-1 rounded-lg">
            @for (m of models; track m.id) {
              <button
                (click)="selectModel(m.id)"
                class="px-3 py-1.5 text-xs font-medium rounded-md transition-all"
                [class]="api.activeModel() === m.id 
                  ? 'bg-white text-text-primary shadow-sm' 
                  : 'text-text-muted hover:text-text-secondary'"
              >
                {{ m.label }}
              </button>
            }
          </div>
        </div>
      </header>

      <!-- Sohbet -->
      <div class="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin" #chatContainer>

        @for (msg of api.chatHistory(); track msg.id) {
          <div class="flex gap-3" [class]="msg.role === 'user' ? 'flex-row-reverse' : ''">
            <!-- Avatar -->
            <div class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                 [class]="msg.role === 'user' ? 'bg-primary-100' : 'bg-primary-50'">
              @if (msg.role === 'user') {
                <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                        d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/>
                </svg>
              } @else {
                <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                        d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
                </svg>
              }
            </div>
            
            <!-- Mesaj -->
            <div class="max-w-[70%] flex flex-col gap-2">
              <div class="rounded-xl p-3"
                   [class]="msg.role === 'user' 
                     ? 'bg-primary-600 text-white rounded-tr-none' 
                     : 'bg-white border border-border-subtle rounded-tl-none'">
                <p class="text-sm whitespace-pre-wrap">{{ msg.content }}</p>
              </div>
              
              @if (msg.role === 'assistant' && msg.model) {
                <div class="flex items-center gap-3 text-xs text-text-muted">
                  <span class="px-2 py-0.5 bg-primary-50 rounded text-primary-600 font-medium">
                    {{ msg.model }}
                  </span>
                  @if (msg.confidence) {
                    <span>{{ (msg.confidence * 100).toFixed(0) }}%</span>
                  }
                  @if (msg.responseTime) {
                    <span>{{ msg.responseTime.toFixed(0) }}ms</span>
                  }
                </div>
                
                @if (msg.context) {
                  <app-context-card [context]="msg.context" [confidence]="msg.confidence || 0" />
                }
              }
            </div>
          </div>
        }

        @if (api.loading()) {
          <div class="flex gap-3">
            <div class="w-8 h-8 rounded-full bg-primary-50 flex items-center justify-center">
              <svg class="w-4 h-4 text-primary-600 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
              </svg>
            </div>
            <div class="bg-white border border-border-subtle rounded-xl rounded-tl-none p-3">
              <div class="flex items-center gap-2 text-sm text-text-muted">
                <span class="w-2 h-2 bg-primary-400 rounded-full animate-bounce"></span>
                <span class="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style="animation-delay: 150ms"></span>
                <span class="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style="animation-delay: 300ms"></span>
              </div>
            </div>
          </div>
        }

        @if (api.error()) {
          <div class="bg-red-50 border border-red-200 rounded-lg p-3">
            <p class="text-sm text-red-600">{{ api.error() }}</p>
          </div>
        }
      </div>

      <!-- Input -->
      <footer class="bg-white border-t border-border-subtle p-4">
        <form (submit)="send($event)" class="flex gap-3">
          <input
            type="text"
            [(ngModel)]="message"
            name="message"
            class="input flex-1"
            placeholder="Sorunuzu yazın..."
            [disabled]="api.loading()"
          />
          <button type="submit" class="btn btn-primary" [disabled]="!message.trim() || api.loading()">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
            </svg>
          </button>
          <button type="button" (click)="api.clearChat()" class="btn btn-secondary" title="Temizle">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
            </svg>
          </button>
        </form>
      </footer>
    </div>
  `
})
export class ChatComponent implements AfterViewChecked, OnInit {
  @ViewChild('chatContainer') chatContainer!: ElementRef;
  
  message = '';
  selectedAgentId = signal<string>('');
  agents = signal<Agent[]>([]);
  private shouldScroll = false;
  
  models: { id: ModelType; label: string }[] = [
    { id: 'GPT', label: 'GPT' },
    { id: 'BERT-CASED', label: 'BERT Cased' },
    { id: 'BERT-SENTIMENT', label: 'BERT Sentiment' }
  ];

  constructor(public api: ApiService) {}

  ngOnInit(): void {
    // agentları yükle
    this.api.loadAgents().subscribe({
      next: () => {
        this.agents.set(this.api.agents());
      }
    });
  }

  // agent değiştiğinde index'i güncelle
  onAgentChange(): void {
    const agentId = this.selectedAgentId();
    if (agentId) {
      const agent = this.agents().find(a => a.id === agentId);
      if (agent) {
        this.api.setActiveIndex(agent.indexName);
      }
    } else {
      this.api.setActiveIndex('default');
    }
  }

  // aktif index adını döndür
  currentIndexName(): string {
    const agentId = this.selectedAgentId();
    if (agentId) {
      const agent = this.agents().find(a => a.id === agentId);
      return agent?.indexName || this.api.activeIndex();
    }
    return this.api.activeIndex();
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  selectModel(m: ModelType): void {
    this.api.setModel(m);
  }

  send(e: Event): void {
    e.preventDefault();
    const text = this.message.trim();
    if (!text) return;

    this.message = '';
    this.shouldScroll = true;

    // agentId varsa gönder
    const agentId = this.selectedAgentId();
    this.api.sendMessage(text, agentId || undefined).subscribe({
      next: () => this.shouldScroll = true,
      error: () => this.shouldScroll = true
    });
  }

  private scrollToBottom(): void {
    try {
      this.chatContainer.nativeElement.scrollTop = this.chatContainer.nativeElement.scrollHeight;
    } catch {}
  }
}
