/*
 * API servisi - backend ile haberleşme
 */

import { Injectable, signal, computed } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, catchError, map, throwError, tap } from 'rxjs';

// tipler

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  confidence?: number;
  context?: string;
  responseTime?: number;
  timestamp: Date;
}

export interface ChatResponse {
  success: boolean;
  data: {
    answer: string;
    context: string;
    all_contexts?: string[];
    confidence: number;
    model_used: string;
    response_time_ms: number;
  };
}

export interface EvaluationResult {
  model: string;
  avg_cosine: number;
  avg_rouge?: number;
  avg_f1?: number;
  accuracy: number;
}

export interface Agent {
  id: string;
  companyId: string;
  name: string;
  description: string;
  indexName: string;
  embeddingModel: string;
  createdAt: string;
  status: 'active' | 'inactive';
}

export interface Company {
  id: string;
  name: string;
  description?: string;
  phone?: string;
  email?: string;
  username: string;
  password?: string; // sadece oluşturulurken göster
  createdAt: string;
}

export interface CurrentUser {
  id: string;
  username: string;
  companyId?: string;
  companyName?: string;
  isSuperAdmin: boolean;
}

export interface IndexInfo {
  name: string;
  documentCount: number;
}

export type ModelType = 'GPT' | 'BERT-CASED' | 'BERT-SENTIMENT';
export type EmbeddingModel = 'text-embedding-3-large' | 'paraphrase-multilingual-MiniLM-L12-v2';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly apiUrl = '/api';
  
  // state
  private _loading = signal(false);
  private _chatHistory = signal<ChatMessage[]>([]);
  private _activeModel = signal<ModelType>('GPT');
  private _evaluationResults = signal<Record<string, EvaluationResult> | null>(null);
  private _error = signal<string | null>(null);
  private _currentUser = signal<CurrentUser | null>(null);
  private _agents = signal<Agent[]>([]);
  private _companies = signal<Company[]>([]);
  private _indices = signal<IndexInfo[]>([]);
  private _activeIndex = signal<string>('default');
  
  // computed
  readonly loading = computed(() => this._loading());
  readonly chatHistory = computed(() => this._chatHistory());
  readonly activeModel = computed(() => this._activeModel());
  readonly evaluationResults = computed(() => this._evaluationResults());
  readonly error = computed(() => this._error());
  readonly currentUser = computed(() => this._currentUser());
  readonly agents = computed(() => this._agents());
  readonly companies = computed(() => this._companies());
  readonly indices = computed(() => this._indices());
  readonly activeIndex = computed(() => this._activeIndex());
  
  readonly isLoggedIn = computed(() => this._currentUser() !== null);
  readonly isSuperAdmin = computed(() => this._currentUser()?.isSuperAdmin === true);

  constructor(private http: HttpClient) {
    // localStorage'dan user yükle
    const saved = localStorage.getItem('currentUser');
    if (saved) {
      try {
        this._currentUser.set(JSON.parse(saved));
      } catch {}
    }
  }

  // ---- model & index ----

  setModel(model: ModelType): void {
    this._activeModel.set(model);
  }

  setActiveIndex(indexName: string): void {
    this._activeIndex.set(indexName);
  }

  loadIndices(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/indices`).pipe(
      tap(r => r.success && this._indices.set(r.data.indices))
    );
  }

  // ---- upload ----

  uploadFile(file: File, indexName: string, embeddingModel: EmbeddingModel): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('indexName', indexName);
    formData.append('embeddingModel', embeddingModel);

    this._loading.set(true);
    return this.http.post<any>(`${this.apiUrl}/upload`, formData).pipe(
      tap(() => { this._loading.set(false); this.loadIndices().subscribe(); }),
      catchError(e => { this._loading.set(false); return throwError(() => new Error(this.handleError(e))); })
    );
  }

  uploadUrl(url: string, indexName: string, embeddingModel: EmbeddingModel): Observable<any> {
    this._loading.set(true);
    return this.http.post<any>(`${this.apiUrl}/upload/url`, { url, indexName, embeddingModel }).pipe(
      tap(() => { this._loading.set(false); this.loadIndices().subscribe(); }),
      catchError(e => { this._loading.set(false); return throwError(() => new Error(this.handleError(e))); })
    );
  }

  // ---- chat ----

  clearChat(): void {
    this._chatHistory.set([]);
    this._error.set(null);
  }

  sendMessage(query: string, agentId?: string): Observable<ChatResponse> {
    const userMsg: ChatMessage = {
      id: this.genId(),
      role: 'user',
      content: query,
      timestamp: new Date()
    };

    this._chatHistory.update(h => [...h, userMsg]);
    this._loading.set(true);
    this._error.set(null);

    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, {
      query,
      model: this._activeModel(),
      indexName: this._activeIndex(),
      agentId: agentId || null
    }).pipe(
      tap(res => {
        const assistantMsg: ChatMessage = {
          id: this.genId(),
          role: 'assistant',
          content: res.data.answer,
          model: res.data.model_used,
          confidence: res.data.confidence,
          context: res.data.context,
          responseTime: res.data.response_time_ms,
          timestamp: new Date()
        };
        this._chatHistory.update(h => [...h, assistantMsg]);
        this._loading.set(false);
      }),
      catchError(e => {
        this._loading.set(false);
        const msg = this.handleError(e);
        this._error.set(msg);
        return throwError(() => new Error(msg));
      })
    );
  }

  // ---- benchmark ----

  runBenchmark(): Observable<any> {
    this._loading.set(true);
    return this.http.post<any>(`${this.apiUrl}/benchmark`, { indexName: this._activeIndex() }).pipe(
      tap(r => { r.data?.results && this._evaluationResults.set(r.data.results); this._loading.set(false); }),
      catchError(e => { this._loading.set(false); return throwError(() => new Error(this.handleError(e))); })
    );
  }

  getEvaluationResults(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/benchmark/results`).pipe(
      tap(r => this._evaluationResults.set(r.data))
    );
  }

  getPlots(): Observable<{ plots: { name: string; path: string }[] }> {
    return this.http.get<any>(`${this.apiUrl}/plots`).pipe(map(r => r.data));
  }

  // ---- auth ----

  login(username: string, password: string): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/auth/login`, { username, password }).pipe(
      tap(r => {
        if (r && r.success && r.data) {
          const user: CurrentUser = {
            id: r.data.userId || 'unknown',
            username: r.data.username || username,
            companyId: r.data.companyId,
            companyName: r.data.companyName,
            isSuperAdmin: r.data.isSuperAdmin || false
          };
          this._currentUser.set(user);
          localStorage.setItem('currentUser', JSON.stringify(user));
          if (r.data.sessionToken) {
            localStorage.setItem('sessionToken', r.data.sessionToken);
          }
        }
      }),
      catchError(error => {
        console.error('Login API error:', error);
        // 404 hatası backend çalışmıyor demektir
        if (error.status === 404 || error.status === 0) {
          return throwError(() => new Error('Backend sunucusu çalışmıyor. Lütfen backend\'i başlatın (node backend/server.js)'));
        }
        return throwError(() => error);
      })
    );
  }

  logout(): void {
    this._currentUser.set(null);
    localStorage.removeItem('currentUser');
    localStorage.removeItem('sessionToken');
  }

  // ---- company (superadmin only) ----

  // public endpoint - company sayısını kontrol et
  checkCompaniesCount(): Observable<{ count: number }> {
    return this.http.get<{ success: boolean; count: number }>(`${this.apiUrl}/companies/count`).pipe(
      map(r => ({ count: r.count || 0 }))
    );
  }

  createCompany(data: { name: string; description?: string; phone?: string; email?: string }): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/admin/companies`, data, this.authHeaders()).pipe(
      tap(r => r.success && this._companies.update(list => [...list, r.data]))
    );
  }

  loadCompanies(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/admin/companies`, this.authHeaders()).pipe(
      tap(r => r.success && this._companies.set(r.data))
    );
  }

  deleteCompany(id: string): Observable<any> {
    return this.http.delete<any>(`${this.apiUrl}/admin/companies/${id}`, this.authHeaders()).pipe(
      tap(r => r.success && this._companies.update(list => list.filter(c => c.id !== id)))
    );
  }

  // ---- agents ----

  createAgent(data: { name: string; description?: string; indexName: string; embeddingModel: EmbeddingModel; dataSourceType?: string; dataSource?: string | null }): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/agents`, data, this.authHeaders()).pipe(
      tap(r => r.success && this._agents.update(list => [...list, r.data]))
    );
  }

  loadAgents(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/agents`, this.authHeaders()).pipe(
      tap(r => r.success && this._agents.set(r.data))
    );
  }

  deleteAgent(id: string): Observable<any> {
    return this.http.delete<any>(`${this.apiUrl}/agents/${id}`, this.authHeaders()).pipe(
      tap(r => r.success && this._agents.update(list => list.filter(a => a.id !== id)))
    );
  }

  // ---- helpers ----

  private authHeaders() {
    const token = localStorage.getItem('sessionToken');
    return { headers: { Authorization: `Bearer ${token}` } };
  }

  private genId(): string {
    return `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  }

  private handleError(error: HttpErrorResponse): string {
    if (error.error?.error) return error.error.error;
    if (error.status === 0) return 'Sunucuya bağlanılamadı';
    if (error.status === 401) return 'Oturum süresi doldu';
    return error.statusText || 'Bilinmeyen hata';
  }
}
