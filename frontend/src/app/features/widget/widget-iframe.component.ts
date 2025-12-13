// widget iframe - müşteri tarafında görünecek chat widget'ı

import { Component, signal, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-widget-iframe',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="h-screen flex flex-col bg-white">
      <!-- chat header -->
      <div class="bg-gradient-to-r from-blue-500 to-teal-500 p-4 text-white">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="font-semibold">{{ agentName() || 'Chatbot' }}</h3>
            <p class="text-xs opacity-90">Size nasıl yardımcı olabilirim?</p>
          </div>
          <button (click)="toggleChat()" class="p-1.5 rounded-full hover:bg-white/20">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
      </div>

      <!-- müşteri bilgi formu - ilk açılışta gösteriliyor -->
      @if (!userInfoSubmitted()) {
        <div class="flex-1 overflow-y-auto p-4 bg-gray-50">
          <div class="max-w-md mx-auto">
            <div class="card p-6">
              <h4 class="text-sm font-semibold text-text-primary mb-1">
                Görüşmeye başlamak için lütfen bilgilerinizi giriniz
              </h4>
              <div class="h-0.5 w-16 bg-primary-600 mb-4"></div>
              
              <form (submit)="submitUserInfo($event)" class="space-y-4">
                <!-- ad soyad -->
                <div>
                  <input
                    type="text"
                    [(ngModel)]="userForm.name"
                    name="name"
                    class="input"
                    placeholder="Ad Soyad"
                    required
                    pattern="[a-zA-ZçğıöşüÇĞIİÖŞÜ\s]{2,50}"
                  />
                  @if (userForm.name && !isValidName()) {
                    <p class="text-xs text-red-600 mt-1">
                      Geçerli bir ad soyad giriniz (sadece harf ve boşluk, 2-50 karakter)
                    </p>
                  }
                </div>

                <!-- email -->
                <div>
                  <input
                    type="email"
                    [(ngModel)]="userForm.email"
                    name="email"
                    class="input"
                    placeholder="E-posta"
                    required
                  />
                </div>

                <!-- telefon -->
                <div>
                  <input
                    type="tel"
                    [(ngModel)]="userForm.phone"
                    name="phone"
                    class="input"
                    placeholder="Telefon"
                    required
                  />
                </div>

                <!-- cinsiyet -->
                <div>
                  <div class="flex gap-4">
                    <label class="flex items-center gap-2 cursor-pointer">
                      <input type="radio" [(ngModel)]="userForm.gender" name="gender" value="male" class="w-4 h-4" />
                      <span class="text-sm text-text-primary">Erkek</span>
                    </label>
                    <label class="flex items-center gap-2 cursor-pointer">
                      <input type="radio" [(ngModel)]="userForm.gender" name="gender" value="female" class="w-4 h-4" />
                      <span class="text-sm text-text-primary">Kadın</span>
                    </label>
                  </div>
                </div>

                <!-- kvkk -->
                <div>
                  <label class="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" [(ngModel)]="userForm.kvkkAccepted" name="kvkk" class="w-4 h-4" required />
                    <span class="text-xs text-text-secondary">
                      <span class="underline">KVKK Aydınlatma Metni</span>'ni okudum ve kabul ediyorum
                    </span>
                  </label>
                </div>

                <!-- gönder -->
                <button
                  type="submit"
                  [disabled]="!canSubmit()"
                  class="btn btn-primary w-full justify-center"
                  [class.opacity-50]="!canSubmit()"
                >
                  Görüşmeyi Başlat
                </button>
              </form>
            </div>
          </div>
        </div>
      }

      <!-- chat arayüzü - form doldurulduktan sonra -->
      @if (userInfoSubmitted()) {
        <div class="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin" #chatContainer>
          @for (msg of messages(); track msg.id) {
            <div class="flex gap-2" [class]="msg.role === 'user' ? 'flex-row-reverse' : ''">
              <div class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                   [class]="msg.role === 'user' ? 'bg-primary-500' : 'bg-gray-200'">
                @if (msg.role === 'user') {
                  <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/>
                  </svg>
                } @else {
                  <svg class="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
                  </svg>
                }
              </div>
              <div class="max-w-[75%]">
                <div class="rounded-lg p-3"
                     [class]="msg.role === 'user' 
                       ? 'bg-primary-500 text-white' 
                       : 'bg-gray-100 text-gray-800'">
                  <p class="text-sm whitespace-pre-wrap">{{ msg.content }}</p>
                </div>
              </div>
            </div>
          }

          @if (loading()) {
            <div class="flex gap-2">
              <div class="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                <svg class="w-4 h-4 text-gray-600 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                        d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
                </svg>
              </div>
              <div class="bg-gray-100 rounded-lg p-3">
                <div class="flex gap-1">
                  <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></span>
                  <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 150ms"></span>
                  <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 300ms"></span>
                </div>
              </div>
            </div>
          }
        </div>

        <!-- input -->
        <div class="border-t border-gray-200 p-4 bg-white">
          <form (submit)="sendMessage($event)" class="flex gap-2">
            <input
              type="text"
              [(ngModel)]="currentMessage"
              name="message"
              class="input flex-1"
              placeholder="Mesajınızı yazın..."
              [disabled]="loading()"
            />
            <button type="submit" class="btn btn-primary" [disabled]="!currentMessage.trim() || loading()">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
              </svg>
            </button>
          </form>
        </div>
      }
    </div>
  `
})
export class WidgetIframeComponent implements OnInit, AfterViewChecked {
  @ViewChild('chatContainer') chatContainer!: ElementRef;
  
  agentId = signal<string>('');
  agentName = signal<string>('');
  userInfoSubmitted = signal(false);
  messages = signal<any[]>([]);
  loading = signal(false);
  currentMessage = '';
  private shouldScroll = false;

  // müşteri bilgi formu
  userForm = {
    name: '',
    email: '',
    phone: '',
    gender: '',
    kvkkAccepted: false
  };

  constructor(
    private route: ActivatedRoute,
    private http: HttpClient
  ) {}

  ngOnInit(): void {
    // query param'dan agent id'yi al
    const id = this.route.snapshot.queryParams['agentId'];
    if (id) {
      this.agentId.set(id);
      this.loadAgent(id);
    }
  }

  ngAfterViewChecked(): void {
    // mesaj geldiğinde scroll yap
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  // agent bilgisini yükle
  loadAgent(id: string): void {
    this.http.get<any>(`/api/agents/${id}`).subscribe({
      next: (res) => {
        if (res.success && res.data) {
          this.agentName.set(res.data.name || 'Chatbot');
        }
      },
      error: () => {
        this.agentName.set('Chatbot');
      }
    });
  }

  // ad soyad validasyonu - sadece harf ve boşluk, 2-50 karakter
  isValidName(): boolean {
    const pattern = /^[a-zA-ZçğıöşüÇĞIİÖŞÜ\s]{2,50}$/;
    return pattern.test(this.userForm.name);
  }

  // form gönderilebilir mi kontrol et
  canSubmit(): boolean {
    return this.userForm.name.trim() !== '' &&
           this.isValidName() &&
           this.userForm.email.trim() !== '' &&
           this.userForm.phone.trim() !== '' &&
           this.userForm.gender !== '' &&
           this.userForm.kvkkAccepted;
  }

  // müşteri bilgilerini gönder ve chat'i başlat
  submitUserInfo(e: Event): void {
    e.preventDefault();
    if (!this.canSubmit()) return;
    
    this.userInfoSubmitted.set(true);
    
    // hoş geldin mesajı
    this.messages.set([{
      id: '1',
      role: 'assistant',
      content: `Merhaba ${this.userForm.name}! Size nasıl yardımcı olabilirim?`
    }]);
  }

  // mesaj gönder
  sendMessage(e: Event): void {
    e.preventDefault();
    const text = this.currentMessage.trim();
    if (!text) return;

    // kullanıcı mesajını ekle
    this.messages.update(msgs => [...msgs, {
      id: Date.now().toString(),
      role: 'user',
      content: text
    }]);

    this.currentMessage = '';
    this.loading.set(true);
    this.shouldScroll = true;

    // API'ye gönder
    this.http.post<any>('/api/chat', {
      query: text,
      agentId: this.agentId(),
      model: 'GPT'
    }).subscribe({
      next: (res) => {
        // cevabı ekle
        this.messages.update(msgs => [...msgs, {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: res.data?.answer || 'Yanıt alınamadı'
        }]);
        this.loading.set(false);
        this.shouldScroll = true;
      },
      error: () => {
        this.loading.set(false);
        this.shouldScroll = true;
      }
    });
  }

  // widget'ı kapat - parent window'a mesaj gönder
  toggleChat(): void {
    if (window.parent) {
      window.parent.postMessage({ type: 'close-widget' }, '*');
    }
  }

  // chat container'ın en altına scroll yap
  private scrollToBottom(): void {
    try {
      this.chatContainer.nativeElement.scrollTop = this.chatContainer.nativeElement.scrollHeight;
    } catch {}
  }
}
