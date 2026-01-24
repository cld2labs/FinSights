import { Link, NavLink } from 'react-router-dom';
import { Menu, X, TrendingUp } from 'lucide-react';
import { useState } from 'react';

export const Header = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'Generate', href: '/generate' },
  ];

  return (
    <header className="sticky top-0 z-50 w-full backdrop-blur">
      {/* Background layer to match hero */}
      <div className="absolute inset-0 -z-10 bg-gradient-to-b from-primary-50/80 via-white/40 to-white/10" />

      {/* Subtle divider */}
      <div className="absolute inset-x-0 bottom-0 h-px bg-gray-200/50" />

      <nav className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Brand */}
          <Link
            to="/"
            className="flex items-center gap-3 transition-opacity hover:opacity-90"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary-600 to-secondary-600 text-white shadow-sm">
              <TrendingUp className="h-5 w-5" />
            </div>

            <div className="leading-tight">
              <div className="text-base font-bold text-gray-900">
                FinSights
              </div>
              <div className="text-xs text-gray-500">
                Intelligent market briefs
              </div>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden items-center gap-1 md:flex">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) =>
                  `rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
                    isActive
                      ? 'text-primary-700 bg-primary-100/60'
                      : 'text-gray-700 hover:text-gray-900 hover:bg-white/40'
                  }`
                }
              >
                {item.name}
              </NavLink>
            ))}
          </div>

          {/* Mobile menu button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden inline-flex items-center justify-center rounded-lg p-2 text-gray-700 hover:bg-white/50"
            aria-label="Toggle menu"
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </button>
        </div>

        {/* Mobile Navigation */}
        <div
          className={`md:hidden overflow-hidden transition-all duration-300 ${
            mobileMenuOpen ? 'max-h-64 pb-4' : 'max-h-0'
          }`}
        >
          <div className="mt-2 rounded-2xl bg-white/80 backdrop-blur shadow-sm ring-1 ring-gray-200 p-2">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  `block rounded-xl px-4 py-3 text-sm font-semibold transition-colors ${
                    isActive
                      ? 'text-primary-700 bg-primary-100/60'
                      : 'text-gray-700 hover:bg-gray-50'
                  }`
                }
              >
                {item.name}
              </NavLink>
            ))}
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;
