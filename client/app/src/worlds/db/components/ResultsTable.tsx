// Renders the last RETURN'd value as a table — the generic shape query
// results actually take (a list of records with the same fields), not
// something specific to any one entity. Anything that isn't a list of
// plain objects (a bare count, a single OBJ, null) falls back to JSON.
export default function ResultsTable({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <p className="results-empty">Ask a question to see results here.</p>;
  }
  if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'object' && value[0] !== null) {
    const rows = value as Record<string, unknown>[];
    const columns = Object.keys(rows[0]);
    return (
      <div className="results-table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c}>{formatCell(row[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  if (Array.isArray(value) && value.length === 0) {
    return <p className="results-empty">No results.</p>;
  }
  return <pre className="results-raw">{JSON.stringify(value, null, 2)}</pre>;
}

function formatCell(v: unknown): string {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  return String(v);
}
