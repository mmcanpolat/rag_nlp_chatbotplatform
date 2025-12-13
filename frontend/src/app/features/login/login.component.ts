// giriş sayfası

import { Component, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, ActivatedRoute } from '@angular/router';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="min-h-screen flex items-center justify-center bg-surface-main p-4">
      <div class="w-full max-w-md page-enter">
        <!-- Logo -->
        <div class="text-center mb-8">
          <div class="w-14 h-14 rounded-xl bg-primary-800 flex items-center justify-center mx-auto mb-4">
            <svg class="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
            </svg>
          </div>
          <h1 class="text-xl font-semibold text-text-primary">RAG Platform</h1>
          <p class="text-sm text-text-muted mt-1">Giriş Yap</p>
        </div>

        <!-- Form -->
        <div class="card p-6">
          <form (submit)="handleLogin($event)" class="space-y-4">
            <!-- Username -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Kullanıcı Adı
              </label>
              <input
                type="text"
                [(ngModel)]="username"
                name="username"
                class="input"
                placeholder="Kullanıcı adınız"
                required
              />
            </div>

            <!-- Şifre -->
            <div>
              <label class="block text-sm font-medium text-text-secondary mb-1.5">
                Şifre
              </label>
              <input
                type="password"
                [(ngModel)]="password"
                name="password"
                class="input"
                placeholder="••••••••"
                required
              />
            </div>

            <!-- Başarı Mesajı -->
            @if (successMessage()) {
              <div class="p-3 bg-green-50 border border-green-200 rounded-lg">
                <p class="text-sm text-green-600 mb-2">{{ successMessage() }}</p>
                @if (route.snapshot.queryParams['password']) {
                  <div class="mt-2 p-2 bg-white rounded border border-green-200">
                    <p class="text-xs text-text-muted mb-1">Kullanıcı Adı: <span class="font-mono font-semibold">{{ username || route.snapshot.queryParams['username'] }}</span></p>
                    <p class="text-xs text-text-muted">Şifre: <span class="font-mono font-semibold">{{ route.snapshot.queryParams['password'] }}</span></p>
                    <p class="text-xs text-red-600 mt-1">⚠️ Bu şifreyi kaydedin, tekrar gösterilmeyecek!</p>
                  </div>
                }
              </div>
            }

            <!-- Hata -->
            @if (error()) {
              <div class="p-3 bg-red-50 border border-red-200 rounded-lg">
                <p class="text-sm text-red-600">{{ error() }}</p>
              </div>
            }

            <!-- Giriş Butonu -->
            <button
              type="submit"
              [disabled]="loading()"
              class="btn btn-primary w-full justify-center"
            >
              @if (loading()) {
                <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
              }
              <span>Giriş Yap</span>
            </button>
          </form>

          <!-- SuperAdmin Bilgileri -->
          <div class="mt-6 pt-6 border-t border-border-subtle">
            <p class="text-xs text-text-muted mb-2 text-center">SuperAdmin Girişi</p>
            <div class="bg-primary-50 p-3 rounded-lg">
              <p class="text-xs text-text-muted mb-1">
                <span class="font-medium">Kullanıcı:</span> 
                <span class="font-mono text-primary-700">admin&#64;ragplatform.com</span>
              </p>
              <p class="text-xs text-text-muted">
                <span class="font-medium">Şifre:</span> 
                <span class="font-mono text-primary-700">Admin123!&#64;#</span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  `
})
export class LoginComponent implements OnInit {
  loading = signal(false);
  error = signal<string | null>(null);
  successMessage = signal<string | null>(null);
  
  username = '';
  password = '';

  constructor(
    private api: ApiService,
    private router: Router,
    public route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    // eğer zaten giriş yapılmışsa yönlendir
    if (this.api.isLoggedIn()) {
      const user = this.api.currentUser();
      if (user?.isSuperAdmin) {
        this.router.navigate(['/admin/companies']);
      } else {
        this.router.navigate(['/agents']);
      }
      return;
    }

    // query params'tan mesaj, username ve password al
    this.route.queryParams.subscribe(params => {
      if (params['message']) {
        this.successMessage.set(params['message']);
      }
      if (params['username']) {
        this.username = params['username'];
      }
      // password varsa göster (sadece ilk oluşturmada)
      if (params['password']) {
        // password'ü bir yerde göster veya otomatik doldur
        // güvenlik için password'ü input'a otomatik doldurmayalım, sadece mesajda belirtelim
      }
    });
  }

  handleLogin(e: Event): void {
    e.preventDefault();
    this.error.set(null);
    this.loading.set(true);

    this.api.login(this.username.trim(), this.password).subscribe({
      next: (res) => {
        this.loading.set(false);
        if (res && res.success) {
          // başarılı giriş
          if (res.data?.isSuperAdmin) {
            this.router.navigate(['/admin/companies']);
          } else {
            this.router.navigate(['/agents']);
          }
        } else {
          this.error.set(res?.error || 'Giriş başarısız');
        }
      },
      error: (err) => {
        this.loading.set(false);
        console.error('Login error:', err);
        
        // 404 veya 0 (network error) - backend çalışmıyor
        if (err?.status === 404 || err?.status === 0) {
          this.error.set('Backend sunucusu çalışmıyor. Lütfen backend\'i başlatın: npm start veya node backend/server.js');
        } else {
          // HTTP hatası veya network hatası
          const errorMessage = err?.error?.error || err?.message || 'Giriş başarısız. Lütfen tekrar deneyin.';
          this.error.set(errorMessage);
        }
      }
    });
  }
}
