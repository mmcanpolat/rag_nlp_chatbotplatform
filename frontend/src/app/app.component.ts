// ana component - sidebar + router outlet

import { Component, signal, OnInit } from '@angular/core';
import { RouterOutlet, Router, NavigationEnd } from '@angular/router';
import { SidebarComponent } from './shared/components/sidebar.component';
import { CommonModule } from '@angular/common';
import { filter } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, SidebarComponent, CommonModule],
  template: `
    <!-- Welcome ve Login sayfalarında sidebar gösterme -->
    @if (showSidebar()) {
      <div class="flex h-screen overflow-hidden bg-surface-main">
        <!-- Yan Menü -->
        <app-sidebar />
        
        <!-- Ana İçerik Alanı -->
        <main class="flex-1 overflow-auto">
          <router-outlet />
        </main>
      </div>
    } @else {
      <!-- Tam ekran (welcome/login) -->
      <router-outlet />
    }
  `,
  styles: [`
    :host {
      display: block;
      height: 100vh;
    }
  `]
})
export class AppComponent implements OnInit {
  title = 'RAG SaaS Platform';
  showSidebar = signal(false);

  constructor(private router: Router) {}

  ngOnInit(): void {
    // ilk yüklemede kontrol et
    this.updateSidebarVisibility();

    // route değişikliklerini dinle
    this.router.events.pipe(
      filter(event => event instanceof NavigationEnd)
    ).subscribe(() => {
      this.updateSidebarVisibility();
    });
  }

  private updateSidebarVisibility(): void {
    const url = this.router.url;
    // welcome ve login sayfalarında sidebar gösterme
    const shouldShow = !url.startsWith('/welcome') && !url.startsWith('/login') && url !== '/';
    this.showSidebar.set(shouldShow);
  }
}
