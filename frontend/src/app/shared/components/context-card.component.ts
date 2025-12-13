/*
 * Context kartı - RAG'ın bulduğu bilgiyi gösteriyor
 */

import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-context-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="mt-2 bg-primary-50 border border-primary-100 rounded-lg p-3">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs font-medium text-primary-600 flex items-center gap-1.5">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
          Kullanılan Bağlam
        </span>
        
        <!-- Güven Göstergesi -->
        <div class="flex items-center gap-1.5">
          <div class="w-16 h-1.5 bg-primary-100 rounded-full overflow-hidden">
            <div 
              class="h-full bg-primary-500 rounded-full transition-all"
              [style.width.%]="confidence * 100"
            ></div>
          </div>
          <span class="text-xs text-primary-600">{{ (confidence * 100).toFixed(0) }}%</span>
        </div>
      </div>
      
      <!-- Bağlam Metni -->
      <p class="text-xs text-text-secondary leading-relaxed">
        {{ isExpanded ? context : truncatedContext }}
      </p>
      
      <!-- Genişlet/Daralt Butonu -->
      @if (context.length > 150) {
        <button
          (click)="toggleExpand()"
          class="mt-2 text-xs text-primary-600 hover:text-primary-700 font-medium"
        >
          {{ isExpanded ? 'Daralt' : 'Devamını oku' }}
        </button>
      }
    </div>
  `
})
export class ContextCardComponent {
  @Input() context: string = '';
  @Input() confidence: number = 0;
  
  isExpanded = false;
  
  get truncatedContext(): string {
    if (this.context.length <= 150) return this.context;
    return this.context.slice(0, 150) + '...';
  }
  
  toggleExpand(): void {
    this.isExpanded = !this.isExpanded;
  }
}
