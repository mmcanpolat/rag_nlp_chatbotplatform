// Angular app entry point

import { bootstrapApplication } from '@angular/platform-browser';
import { provideRouter, Routes } from '@angular/router';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideAnimations } from '@angular/platform-browser/animations';

import { AppComponent } from './app/app.component';
import { LoginComponent } from './app/features/login/login.component';
import { ChatComponent } from './app/features/chat/chat.component';
import { AnalyticsComponent } from './app/features/analytics/analytics.component';
import { AgentsComponent } from './app/features/agents/agents.component';
import { CompaniesComponent } from './app/features/admin/companies.component';
import { WidgetEmbedComponent } from './app/features/widget/widget-embed.component';
import { WidgetIframeComponent } from './app/features/widget/widget-iframe.component';
import { authGuard, superAdminGuard } from './app/core/guards/auth.guard';

const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  { path: 'chat', component: ChatComponent, canActivate: [authGuard] },
  { path: 'analytics', component: AnalyticsComponent, canActivate: [authGuard] },
  { path: 'agents', component: AgentsComponent, canActivate: [authGuard] },
  { path: 'admin/companies', component: CompaniesComponent, canActivate: [superAdminGuard] },
  { path: 'widget/embed/:id', component: WidgetEmbedComponent, canActivate: [authGuard] },
  { path: 'widget/iframe', component: WidgetIframeComponent },
  { path: '**', redirectTo: 'login' }
];

bootstrapApplication(AppComponent, {
  providers: [
    provideRouter(routes),
    provideHttpClient(withFetch()),
    provideAnimations()
  ]
}).catch(err => console.error(err));
