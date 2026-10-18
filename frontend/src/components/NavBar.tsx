"use client";

/** Top navigation between the evaluation page and the runs history. */

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Evaluate" },
  { href: "/runs", label: "Runs" },
];

export default function NavBar() {
  const pathname = usePathname();
  return (
    <nav className="nav">
      <span className="nav-title">Global-MMLU eval</span>
      {LINKS.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          className={pathname === link.href ? "nav-link active" : "nav-link"}
        >
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
