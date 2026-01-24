import { Link } from 'react-router-dom';
import {
  ArrowRight,
  FileText,
  Sparkles,
  Zap,
  ShieldCheck,
  TrendingUp,
  BarChart3,
} from 'lucide-react';

export const Home = () => {
  const features = [
    {
      icon: FileText,
      title: 'Financial Document Support',
      description: 'Upload earnings transcripts, annual reports, research notes, and PDFs.',
    },
    {
      icon: Sparkles,
      title: 'AI-Powered Market Briefs',
      description: 'Get concise, actionable insights tailored for financial analysis.',
    },
    {
      icon: BarChart3,
      title: 'Key Metrics Highlighting',
      description: 'Surfaces key numbers, guidance, risks, and catalysts automatically.',
    },
    {
      icon: Zap,
      title: 'Fast and Efficient',
      description: 'Generate a brief from a single uploaded document in seconds.',
    },
  ];

  const steps = [
    {
      number: 1,
      title: 'Upload a Document',
      description: 'Upload one report, transcript, or research document.',
    },
    {
      number: 2,
      title: 'Generate a Brief',
      description: 'FinSights summarizes and structures the key takeaways.',
    },
    {
      number: 3,
      title: 'Review Insights',
      description: 'Scan trends, risks, and opportunities with confidence.',
    },
  ];

  const highlights = [
    { icon: TrendingUp, label: 'Market brief format' },
    { icon: ShieldCheck, label: 'Designed for clarity' },
    { icon: Zap, label: 'Single doc workflow' },
  ];

  return (
    <div className="relative">
      {/* Background */}
      <div className="absolute inset-0 -z-10">
        <div className="h-[520px] bg-gradient-to-b from-primary-50 via-white to-white" />
        <div className="pointer-events-none absolute inset-0 opacity-40">
          <div className="absolute -top-24 left-1/2 h-64 w-64 -translate-x-1/2 rounded-full bg-primary-200 blur-3xl" />
          <div className="absolute top-24 right-10 h-72 w-72 rounded-full bg-secondary-200 blur-3xl" />
          <div className="absolute top-44 left-10 h-72 w-72 rounded-full bg-primary-100 blur-3xl" />
        </div>
      </div>

      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <div className="space-y-20">
          {/* Hero */}
          <section className="pt-10 sm:pt-14">
            <div className="mx-auto max-w-3xl text-center">
              {/* Brand row */}
              <div className="flex items-center justify-center gap-3">
                <img
                  src="/cloud2labs-logo.png"
                  alt="Cloud2Labs"
                  className="h-12 w-12 rounded-xl bg-white p-2 shadow-sm ring-1 ring-gray-200"
                />
                <span className="text-sm font-semibold text-gray-700">
                  Cloud2 Labs
                </span>
              </div>

              {/* Badge */}
              <div className="mt-5 flex items-center justify-center">
                <span className="inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-semibold text-gray-700 shadow-sm ring-1 ring-gray-200">
                  <TrendingUp className="h-4 w-4 text-primary-600" />
                  FinSights - Financial
                </span>
              </div>

              {/* Title */}
              <h1 className="mt-6 text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl md:text-6xl">
                Intelligent daily
                <span className="block bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
                  market briefs
                </span>
              </h1>

              {/* Subtitle */}
              <p className="mt-5 text-lg leading-8 text-gray-600 sm:text-xl">
                Upload a financial report, earnings transcript, or research document and get a concise,
                actionable brief powered by AI. Built for fast reading and quick decisions.
              </p>

              {/* Highlights */}
              <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                {highlights.map((h, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm ring-1 ring-gray-200"
                  >
                    <h.icon className="h-4 w-4 text-primary-600" />
                    {h.label}
                  </span>
                ))}
              </div>

              {/* CTA */}
              <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
                <Link to="/generate" className="w-full sm:w-auto">
                  <button className="btn-primary flex w-full items-center justify-center gap-2 px-8 py-3 text-base sm:text-lg">
                    Generate a brief
                    <ArrowRight className="h-5 w-5" />
                  </button>
                </Link>

                <Link to="/generate" className="w-full sm:w-auto">
                  <button className="w-full rounded-lg bg-white px-8 py-3 text-base font-semibold text-gray-800 shadow-sm ring-1 ring-gray-200 transition-colors hover:bg-gray-50 sm:text-lg">
                    Upload a document
                  </button>
                </Link>
              </div>

              <p className="mt-4 text-sm text-gray-500">
                Upload 1 document. Supported formats depend on backend configuration.
              </p>
            </div>

            {/* Hero Card */}
            {/* <div className="mx-auto mt-10 max-w-5xl">
              <div className="rounded-2xl bg-white p-6 shadow-xl ring-1 ring-gray-200 sm:p-8">
                <div className="grid gap-6 md:grid-cols-3">
                  <div className="rounded-xl bg-gray-50 p-5 ring-1 ring-gray-200">
                    <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                      <FileText className="h-4 w-4 text-primary-600" />
                      Inputs
                    </div>
                    <p className="mt-2 text-sm text-gray-600">
                      Earnings transcripts, annual reports, research notes, and PDFs.
                    </p>
                  </div>

                  <div className="rounded-xl bg-gray-50 p-5 ring-1 ring-gray-200">
                    <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                      <Sparkles className="h-4 w-4 text-primary-600" />
                      Output
                    </div>
                    <p className="mt-2 text-sm text-gray-600">
                      Executive brief, key numbers, risks, and opportunities in a clean layout.
                    </p>
                  </div>

                  <div className="rounded-xl bg-gray-50 p-5 ring-1 ring-gray-200">
                    <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                      <Zap className="h-4 w-4 text-primary-600" />
                      Workflow
                    </div>
                    <p className="mt-2 text-sm text-gray-600">
                      Single document upload, generate, review, and reset.
                    </p>
                  </div>
                </div>
              </div>
            </div> */}
          </section>

          {/* Features */}
          <section>
            <div className="mx-auto max-w-3xl text-center">
              <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
                Why choose FinSights?
              </h2>
              <p className="mt-3 text-lg text-gray-600">
                A focused summarization experience for financial documents, designed to scan fast.
              </p>
            </div>

            <div className="mt-10 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className="group rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200 transition-all hover:-translate-y-0.5 hover:shadow-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary-50 ring-1 ring-primary-100">
                      <feature.icon className="h-6 w-6 text-primary-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {feature.title}
                    </h3>
                  </div>
                  <p className="mt-3 text-sm leading-6 text-gray-600">
                    {feature.description}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* How it works */}
          <section>
            <div className="mx-auto max-w-3xl text-center">
              <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
                How it works
              </h2>
              <p className="mt-3 text-lg text-gray-600">
                Simple flow with a single document upload.
              </p>
            </div>

            <div className="mt-10 grid gap-6 md:grid-cols-3">
              {steps.map((step, index) => (
                <div
                  key={index}
                  className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200"
                >
                  <div className="flex items-center gap-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-primary-600 to-secondary-600 text-white text-lg font-bold shadow-sm">
                      {step.number}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">{step.title}</h3>
                      <p className="mt-1 text-sm text-gray-600">{step.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* CTA */}
          <section className="pb-16">
            <div className="rounded-3xl bg-gradient-to-r from-primary-600 to-secondary-600 p-8 text-center text-white shadow-xl sm:p-12">
              <h2 className="text-3xl font-bold sm:text-4xl">
                Ready to generate your market brief?
              </h2>
              <p className="mx-auto mt-4 max-w-2xl text-lg opacity-90">
                Upload one financial document and get insights in seconds.
              </p>

              <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
                <Link to="/generate" className="w-full sm:w-auto">
                  <button className="w-full rounded-lg bg-white px-8 py-3 text-base font-semibold text-primary-700 shadow-sm transition-colors hover:bg-gray-100 sm:text-lg">
                    Start with FinSights
                  </button>
                </Link>
                <Link to="/generate" className="w-full sm:w-auto">
                  <button className="w-full rounded-lg bg-transparent px-8 py-3 text-base font-semibold text-white ring-1 ring-white/40 transition-colors hover:bg-white/10 sm:text-lg">
                    Go to upload
                  </button>
                </Link>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default Home;
