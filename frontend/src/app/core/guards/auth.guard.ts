// auth guard - oturum açılmadan sayfalara erişimi engeller

import { inject } from '@angular/core';
import { Router, CanActivateFn } from '@angular/router';
import { ApiService } from '../services/api.service';

export const authGuard: CanActivateFn = (route, state) => {
  const apiService = inject(ApiService);
  const router = inject(Router);

  // eğer giriş yapılmışsa erişime izin ver
  if (apiService.isLoggedIn()) {
    return true;
  }

  // giriş yapılmamışsa login sayfasına yönlendir
  router.navigate(['/login'], { queryParams: { returnUrl: state.url } });
  return false;
};

// superadmin guard - sadece superadmin erişebilir
export const superAdminGuard: CanActivateFn = (route, state) => {
  const apiService = inject(ApiService);
  const router = inject(Router);

  if (apiService.isLoggedIn() && apiService.isSuperAdmin()) {
    return true;
  }

  // superadmin değilse ana sayfaya yönlendir
  router.navigate(['/login']);
  return false;
};

