/*
 * Sol yan menü
 */

import { Component, computed } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';

interface NavItem {
  path: string;
  label: string;
  icon: string;
  superAdminOnly?: boolean;
}

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  template: `
    <aside class="w-64 h-full bg-white border-r border-border-subtle flex flex-col">
      <!-- Logo -->
      <div class="p-5 border-b border-border-subtle">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-lg bg-primary-800 flex items-center justify-center">
            <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
            </svg>
          </div>
          <div>
            <h1 class="font-semibold text-text-primary text-sm">RAG Platform</h1>
          </div>
        </div>
      </div>

      <!-- Navigasyon -->
      <nav class="flex-1 p-3 space-y-1">
        <p class="px-3 py-2 text-xs font-medium text-text-muted uppercase tracking-wider">
          Ana Menü
        </p>
        
        @for (item of visibleMenuItems(); track item.path) {
          <a [routerLink]="item.path"
             routerLinkActive="active"
             class="nav-item">
            <span [innerHTML]="item.icon"></span>
            <span>{{ item.label }}</span>
          </a>
        }

        @if (api.isSuperAdmin()) {
          <div class="pt-4">
            <p class="px-3 py-2 text-xs font-medium text-text-muted uppercase tracking-wider">
              Süper Admin
            </p>
            
            @for (item of adminMenuItems; track item.path) {
              <a [routerLink]="item.path"
                 routerLinkActive="active"
                 class="nav-item">
                <span [innerHTML]="item.icon"></span>
                <span>{{ item.label }}</span>
              </a>
            }
          </div>
        }
      </nav>

      <!-- Kullanıcı Bilgisi -->
      @if (api.currentUser()) {
        <div class="p-3 border-t border-border-subtle">
          <div class="flex items-center gap-2.5">
            <div class="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
              <svg class="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/>
              </svg>
            </div>
            <div class="flex-1 min-w-0">
              <p class="text-sm font-medium text-text-primary truncate">
                {{ api.currentUser()?.username }}
              </p>
              <p class="text-xs text-text-muted">
                {{ api.isSuperAdmin() ? 'Süper Admin' : api.currentUser()?.companyName }}
              </p>
            </div>
            <button (click)="logout()" class="p-1.5 rounded hover:bg-primary-50" title="Çıkış">
              <svg class="w-4 h-4 text-text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/>
              </svg>
            </button>
          </div>
        </div>
      }
    </aside>
  `
})
export class SidebarComponent {
  constructor(public api: ApiService) {}

  // normal kullanıcı menüsü
  mainMenuItems: NavItem[] = [
    {
      path: '/chat',
      label: 'Sohbet',
      icon: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
            </svg>`
    },
    {
      path: '/agents',
      label: 'Agent Oluştur',
      icon: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                    d="M12 4v16m8-8H4"/>
            </svg>`
    },
    {
      path: '/analytics',
      label: 'Analiz',
      icon: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>`
    }
  ];

  // superadmin menüsü
  adminMenuItems: NavItem[] = [
    {
      path: '/admin/companies',
      label: 'Şirket Yönetimi',
      icon: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
                    d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"/>
            </svg>`
    }
  ];

  visibleMenuItems = computed(() => {
    if (!this.api.isLoggedIn()) return [];
    return this.mainMenuItems;
  });

  logout() {
    this.api.logout();
    window.location.href = '/login';
  }
}
