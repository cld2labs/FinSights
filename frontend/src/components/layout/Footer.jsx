export const Footer = () => {
  return (
    <footer className="relative mt-16">
      {/* Background */}
      <div className="absolute inset-0 -z-10 bg-gradient-to-t from-primary-50/80 via-white/60 to-white/20" />

      {/* Divider */}
      <div className="absolute inset-x-0 top-0 h-px bg-gray-200/50" />

      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="flex items-center gap-2">
            <img
              src="/cloud2labs-logo.png"
              alt="Cloud2Labs"
              className="h-8 object-contain"
            />
            <span className="text-sm font-medium text-gray-700">
              Cloud2 Labs
            </span>
          </div>

          <p className="text-xs text-gray-500">
            © 2025 Cloud2 Labs. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
