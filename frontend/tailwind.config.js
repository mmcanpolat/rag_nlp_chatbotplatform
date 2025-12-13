// tailwind config - gri tonları tema

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  theme: {
    extend: {
      // Renk paleti - Gri tonları
      colors: {
        // Ana renkler (koyu gri tonları)
        primary: {
          50:  '#F9FAFB',
          100: '#F3F4F6',
          200: '#E5E7EB',
          300: '#D1D5DB',
          400: '#9CA3AF',
          500: '#6B7280',
          600: '#4B5563',
          700: '#374151',
          800: '#1F2937',
          900: '#111827',
        },
        
        // Yüzey renkleri
        surface: {
          main: '#F8FAFC',
          paper: '#FFFFFF',
          overlay: 'rgba(0, 0, 0, 0.5)',
        },
        
        // Metin renkleri
        text: {
          primary: '#1F2937',
          secondary: '#4B5563',
          muted: '#9CA3AF',
          inverse: '#FFFFFF',
        },
        
        // Kenarlık renkleri
        border: {
          subtle: '#E5E7EB',
          default: '#D1D5DB',
          strong: '#9CA3AF',
        },
        
        // Durum renkleri
        status: {
          success: '#059669',
          warning: '#D97706',
          error: '#DC2626',
          info: '#2563EB',
        },
      },
      
      // Yazı tipleri
      fontFamily: {
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        heading: ['Sora', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      
      // Gölgeler
      boxShadow: {
        'soft': '0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06)',
        'medium': '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
        'large': '0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.03)',
      },
      
      // Kenar yuvarlaklıkları
      borderRadius: {
        'xl': '0.875rem',
        '2xl': '1rem',
        '3xl': '1.25rem',
      },
      
      // Animasyonlar
      animation: {
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
      },
      
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
};
