import { SCHEMA } from '../data/schema';

export default function SchemaPanel() {
  return (
    <aside className="schema-rail" aria-label="Database schema">
      <h2>Schema</h2>
      {SCHEMA.map((e) => (
        <div key={e.entity} className="schema-entity">
          <div className="schema-entity-name">{e.entity}</div>
          <ul className="schema-fields">
            {e.fields.map((f) => (
              <li key={f.name}>
                <span className="schema-field-name">{f.name}</span>
                <span className="schema-field-type">{f.type}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </aside>
  );
}
