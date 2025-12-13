// ilk açılış sayfası - hiç company yoksa buraya yönlendiriliyor

import { Component, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-welcome',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    @if (checking()) {
      <div class="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100 flex items-center justify-center">
        <div class="text-center">
          <svg class="w-8 h-8 text-primary-600 animate-spin mx-auto mb-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
          </svg>
          <p class="text-sm text-text-muted">Yükleniyor...</p>
        </div>
      </div>
    } @else {
    <div class="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100 flex items-center justify-center p-6">
      <div class="max-w-2xl w-full">
        <!-- Hoş Geldiniz Kartı -->
        <div class="card p-8 mb-6 text-center">
          <div class="w-16 h-16 rounded-full bg-primary-500 flex items-center justify-center mx-auto mb-4">
            <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
          <h1 class="text-2xl font-bold text-text-primary mb-2">Hoş Geldiniz!</h1>
          <p class="text-sm text-text-muted">
            RAG SaaS Platform'unu kullanmaya başlamak için ilk şirketinizi oluşturun
          </p>
        </div>

        <!-- İlk Şirket Oluşturma Formu -->
        <div class="card p-6">
          <h2 class="text-lg font-semibold text-text-primary mb-4">İlk Şirketi Oluştur</h2>
          
          <form (ngSubmit)="createFirstCompany()" class="space-y-4">
            <!-- Şirket Adı -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Şirket Adı *
              </label>
              <input
                type="text"
                [(ngModel)]="companyForm.name"
                name="name"
                class="input"
                placeholder="Şirket adınızı girin"
                required
              />
            </div>

            <!-- Açıklama -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Açıklama (Opsiyonel)
              </label>
              <textarea
                [(ngModel)]="companyForm.description"
                name="description"
                class="input"
                rows="3"
                placeholder="Şirket hakkında kısa bilgi"
              ></textarea>
            </div>

            <!-- E-posta -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                E-posta (Opsiyonel)
              </label>
              <input
                type="email"
                [(ngModel)]="companyForm.email"
                name="email"
                class="input"
                placeholder="ornek@email.com"
              />
            </div>

            <!-- Telefon -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Telefon (Opsiyonel)
              </label>
              <input
                type="tel"
                [(ngModel)]="companyForm.phone"
                name="phone"
                class="input"
                placeholder="+90 555 123 45 67"
              />
            </div>

            <!-- Hata Mesajı -->
            @if (error()) {
              <div class="bg-red-50 border border-red-200 rounded-lg p-3">
                <p class="text-sm text-red-600">{{ error() }}</p>
              </div>
            }

            <!-- Oluştur Butonu -->
            <button
              type="submit"
              [disabled]="!companyForm.name.trim() || creating()"
              class="btn btn-primary w-full justify-center"
              [class.opacity-50]="!companyForm.name.trim() || creating()"
            >
              @if (creating()) {
                <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                <span>Oluşturuluyor...</span>
              } @else {
                <span>Şirketi Oluştur ve Başla</span>
              }
            </button>
          </form>
        </div>

        <!-- Bilgi Notu -->
        <div class="mt-6 text-center">
          <p class="text-xs text-text-muted">
            Şirket oluşturulduktan sonra otomatik olarak giriş yapacaksınız
          </p>
        </div>
      </div>
    </div>
    }
  `
})
export class WelcomeComponent implements OnInit {
  companyForm = {
    name: '',
    description: '',
    email: '',
    phone: ''
  };
  
  creating = signal(false);
  error = signal<string | null>(null);
  checking = signal(false); // başlangıçta false, API çağrısı başladığında true olacak

  constructor(
    private api: ApiService,
    private router: Router
  ) {}

  ngOnInit(): void {
    // eğer zaten company varsa login'e yönlendir
    // ama önce formu göster, sonra kontrol et
    setTimeout(() => {
      this.checkCompanies();
    }, 100);
  }

  checkCompanies(): void {
    this.checking.set(true);
    // public endpoint ile company sayısını kontrol et
    this.api.checkCompaniesCount().subscribe({
      next: (res) => {
        this.checking.set(false);
        if (res && res.count > 0) {
          // company varsa login'e yönlendir
          setTimeout(() => {
            this.router.navigate(['/login']);
          }, 500);
        }
        // count 0 ise formu göster (zaten gösteriliyor)
      },
      error: (err) => {
        this.checking.set(false);
        console.error('Company kontrolü hatası:', err);
        // hata durumunda da formu göster, kullanıcı company oluşturabilir
        // API çalışmıyorsa bile form gösterilsin
      }
    });
  }

  async createFirstCompany(): Promise<void> {
    if (!this.companyForm.name.trim()) return;

    this.creating.set(true);
    this.error.set(null);

    try {
      // superadmin olarak giriş yap (ilk company için)
      const loginRes = await this.api.login('Admin123', '!AdminPassword123').toPromise();
      
      if (!loginRes?.success) {
        throw new Error('SuperAdmin girişi başarısız');
      }

      // company oluştur
      const companyRes = await this.api.createCompany({
        name: this.companyForm.name,
        description: this.companyForm.description,
        email: this.companyForm.email,
        phone: this.companyForm.phone
      }).toPromise();

      if (!companyRes?.success) {
        throw new Error(companyRes?.error || 'Şirket oluşturulamadı');
      }

      // başarılı - login sayfasına yönlendir
      // company bilgilerini query param olarak gönder
      const companyData = companyRes.data;
      this.router.navigate(['/login'], {
        queryParams: { 
          message: 'Şirket başarıyla oluşturuldu! Aşağıdaki bilgilerle giriş yapabilirsiniz.',
          username: companyData?.username,
          password: companyData?.password  // ilk oluşturmada göster
        }
      });

    } catch (err: any) {
      this.error.set(err?.message || 'Bir hata oluştu');
    } finally {
      this.creating.set(false);
    }
  }
}

