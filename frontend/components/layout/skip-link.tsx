/**
 * The first thing in the tab order: a way past the header.
 *
 * Visually hidden until focused, which is the only form of this that both
 * keyboard users and sighted users are well served by.
 */
export function SkipLink({ targetId = "main-content" }: { targetId?: string }) {
  return (
    <a href={`#${targetId}`} className="skip-link touch-target">
      Skip to main content
    </a>
  );
}
