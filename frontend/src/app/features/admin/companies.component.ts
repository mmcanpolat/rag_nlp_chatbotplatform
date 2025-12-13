/*
 * Şirket yönetimi - sadece superadmin
 */

import { Component, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, Company } from '../../core/services/api.service';

@Component({
  selector: 'app-companies',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="h-full overflow-y-auto scrollbar-thin">
      <div class="p-6 max-w-5xl mx-auto page-enter">
        <!-- Başlık -->
        <header class="mb-6">
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-lg font-semibold text-text-primary">Şirket Yönetimi</h1>
              <p class="text-sm text-text-muted">Şirket oluştur ve yönet</p>
            </div>
            
            <button (click)="showModal.set(true)" class="btn btn-primary">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
              </svg>
              <span>Şirket Oluştur</span>
            </button>
          </div>
        </header>


        <!-- Şirket Listesi -->
        @if (api.companies().length > 0) {
          <div class="space-y-3">
            @for (company of api.companies(); track company.id) {
              <div class="card p-4">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-4">
                    <div class="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center">
                      <svg class="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"/>
                      </svg>
                    </div>
                    <div>
                      <h3 class="text-sm font-medium text-text-primary">{{ company.name }}</h3>
                      <p class="text-xs text-text-muted">
                        Kullanıcı: <span class="font-mono">{{ company.username }}</span>
                      </p>
                    </div>
                  </div>
                  
                  <div class="flex items-center gap-2">
                    @if (company.email) {
                      <span class="text-xs text-text-muted">{{ company.email }}</span>
                    }
                    <button (click)="deleteCompany(company.id)" 
                            class="p-2 text-text-muted hover:text-red-500 rounded">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            }
          </div>
        } @else {
          <div class="card p-8 text-center">
            <div class="w-14 h-14 rounded-xl bg-primary-100 flex items-center justify-center mx-auto mb-4">
              <svg class="w-7 h-7 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                      d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5"/>
              </svg>
            </div>
            <p class="text-sm text-text-muted">Henüz şirket yok. İlk şirketi oluşturun.</p>
          </div>
        }

        <!-- Hata -->
        @if (error()) {
          <div class="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p class="text-sm text-red-600">{{ error() }}</p>
          </div>
        }
      </div>
    </div>

    <!-- Şirket Oluşturma Modal -->
    @if (showModal()) {
      <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50" (click)="closeModal()">
        <div class="bg-white rounded-xl shadow-xl w-full max-w-md mx-4 page-enter" (click)="$event.stopPropagation()">
          <div class="p-5 border-b border-border-subtle">
            <h2 class="text-lg font-semibold text-text-primary">Yeni Şirket</h2>
          </div>
          
          <form (submit)="createCompany($event)" class="p-5 space-y-4">
            <!-- İsim (zorunlu) -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Şirket Adı <span class="text-red-500">*</span>
              </label>
              <input
                type="text"
                [(ngModel)]="form.name"
                name="name"
                class="input"
                placeholder="Şirket adı"
                required
              />
            </div>

            <!-- Açıklama -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Açıklama
              </label>
              <textarea
                [(ngModel)]="form.description"
                name="description"
                class="input"
                rows="2"
                placeholder="Opsiyonel"
              ></textarea>
            </div>

            <!-- Telefon -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Telefon
              </label>
              <input
                type="tel"
                [(ngModel)]="form.phone"
                name="phone"
                class="input"
                placeholder="Opsiyonel"
              />
            </div>

            <!-- Email -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Email
              </label>
              <input
                type="email"
                [(ngModel)]="form.email"
                name="email"
                class="input"
                placeholder="Opsiyonel"
              />
            </div>

            <!-- Butonlar -->
            <div class="flex gap-3 pt-2">
              <button type="button" (click)="closeModal()" class="btn btn-secondary flex-1">
                İptal
              </button>
              <button type="submit" [disabled]="creating() || !form.name.trim()" class="btn btn-primary flex-1">
                @if (creating()) {
                  <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                  </svg>
                }
                <span>Oluştur</span>
              </button>
            </div>
          </form>
        </div>
      </div>
    }

    <!-- Credentials Popup Modal -->
    @if (newCompany()) {
      <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50" (click)="closeCredentialsModal()">
        <div class="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4 page-enter" (click)="$event.stopPropagation()">
          <div class="p-6">
            <!-- Başlık -->
            <div class="flex items-center justify-between mb-4">
              <div>
                <h2 class="text-xl font-semibold text-text-primary">Şirket Oluşturuldu!</h2>
                <p class="text-sm text-text-muted mt-1">Aşağıdaki bilgileri şirketle paylaşın</p>
              </div>
              <button (click)="closeCredentialsModal()" class="p-2 hover:bg-gray-100 rounded-lg">
                <svg class="w-5 h-5 text-text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                </svg>
              </button>
            </div>

            <!-- Uyarı -->
            <div class="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p class="text-xs text-yellow-800">
                ⚠️ Bu bilgiler sadece bir kez gösterilir. Lütfen kaydedin!
              </p>
            </div>

            <!-- Şirket Bilgileri -->
            <div class="space-y-3 mb-4">
              <div class="p-3 bg-gray-50 rounded-lg">
                <p class="text-xs text-text-muted mb-1">Şirket Adı</p>
                <p class="text-sm font-medium text-text-primary">{{ newCompany()?.name }}</p>
              </div>
            </div>

            <!-- Kullanıcı Bilgileri -->
            <div class="space-y-3">
              <!-- Username -->
              <div class="p-4 bg-primary-50 rounded-lg border border-primary-200">
                <div class="flex items-center justify-between mb-2">
                  <p class="text-xs font-medium text-primary-700">Kullanıcı Adı (Email)</p>
                  <button (click)="copy(newCompany()?.username || '')" 
                          class="text-xs text-primary-600 hover:text-primary-800 font-medium">
                    Kopyala
                  </button>
                </div>
                <p class="font-mono text-sm font-semibold text-primary-900 break-all">{{ newCompany()?.username }}</p>
              </div>
              
              <!-- Password -->
              <div class="p-4 bg-red-50 rounded-lg border border-red-200">
                <div class="flex items-center justify-between mb-2">
                  <p class="text-xs font-medium text-red-700">Şifre (24 karakter)</p>
                  <button (click)="copy(newCompany()?.password || '')" 
                          class="text-xs text-red-600 hover:text-red-800 font-medium">
                    Kopyala
                  </button>
                </div>
                <p class="font-mono text-sm font-semibold text-red-900 break-all">{{ newCompany()?.password }}</p>
              </div>
            </div>

            <!-- Buton -->
            <div class="mt-6">
              <button (click)="closeCredentialsModal()" class="btn btn-primary w-full">
                Tamam
              </button>
            </div>
          </div>
        </div>
      </div>
    }
  `
})
export class CompaniesComponent implements OnInit {
  showModal = signal(false);
  creating = signal(false);
  error = signal<string | null>(null);
  newCompany = signal<Company | null>(null);
  
  form = {
    name: '',
    description: '',
    phone: '',
    email: ''
  };

  constructor(public api: ApiService) {}

  ngOnInit(): void {
    this.api.loadCompanies().subscribe();
  }

  createCompany(e: Event): void {
    e.preventDefault();
    if (!this.form.name.trim()) return;

    this.creating.set(true);
    this.error.set(null);

    this.api.createCompany({
      name: this.form.name,
      description: this.form.description || undefined,
      phone: this.form.phone || undefined,
      email: this.form.email || undefined
    }).subscribe({
      next: (res) => {
        this.creating.set(false);
        if (res.success) {
          this.newCompany.set(res.data);
          this.closeModal();
          this.resetForm();
        }
      },
      error: (err) => {
        this.creating.set(false);
        this.error.set(err.message);
      }
    });
  }

  deleteCompany(id: string): void {
    if (!confirm('Bu şirketi silmek istediğinizden emin misiniz?')) return;
    
    this.api.deleteCompany(id).subscribe({
      error: (err) => this.error.set(err.message)
    });
  }

  copy(text: string): void {
    navigator.clipboard.writeText(text);
  }

  closeModal(): void {
    this.showModal.set(false);
  }

  closeCredentialsModal(): void {
    this.newCompany.set(null);
  }

  resetForm(): void {
    this.form = { name: '', description: '', phone: '', email: '' };
  }
}

