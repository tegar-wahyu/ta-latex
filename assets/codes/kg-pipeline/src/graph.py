"""Step D: Neo4j graph construction and querying."""

import logging

import networkx as nx
from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


def get_driver():
    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set")
    uri: str = NEO4J_URI
    user: str = NEO4J_USERNAME or "neo4j"
    pwd: str = NEO4J_PASSWORD
    return GraphDatabase.driver(uri, auth=(user, pwd))


def seed_from_daftar_isi(
    doc_structure,  # DocumentStructure from ingestion.py — avoids circular import
    document_name: str,
    driver,
) -> int:
    """Seed KG skeleton (Document, Bab, SubBab, SubSubBab nodes) from the Daftar Isi.

    Runs *before* extraction so the structural nodes exist even if LLM
    extraction fails or is partial.  All operations use MERGE so re-running
    is idempotent.

    Returns the number of distinct Bab + SubBab + SubSubBab nodes merged.
    """
    if not doc_structure.found or not doc_structure.entries:
        return 0

    count = 0
    with driver.session() as session:
        session.run(
            "MERGE (d:Document {name: $doc_name}) ON CREATE SET d.uploaded_at = timestamp()",
            doc_name=document_name,
        )

        seen_babs: set[str] = set()
        seen_sub_babs: set[tuple[str, str]] = set()  # (bab, sub_bab)
        seen_sub_sub_babs: set[tuple[str, str, str]] = (
            set()
        )  # (bab, sub_bab, sub_sub_bab)

        for entry in doc_structure.entries:
            bab_name = entry.bab

            if bab_name not in seen_babs:
                seen_babs.add(bab_name)
                session.run(
                    """
                    MERGE (b:Bab {name: $bab_name, document: $doc_name})
                    WITH b
                    MATCH (d:Document {name: $doc_name})
                    MERGE (d)-[:hasBab]->(b)
                    """,
                    bab_name=bab_name,
                    doc_name=document_name,
                )
                count += 1

            if entry.sub_bab:
                sub_bab_key = (bab_name, entry.sub_bab)
                if sub_bab_key not in seen_sub_babs:
                    seen_sub_babs.add(sub_bab_key)
                    session.run(
                        """
                        MERGE (sb:SubBab {name: $sub_bab_name, bab: $bab_name, document: $doc_name})
                        WITH sb
                        MATCH (b:Bab {name: $bab_name, document: $doc_name})
                        MERGE (b)-[:hasSubBab]->(sb)
                        """,
                        sub_bab_name=entry.sub_bab,
                        bab_name=bab_name,
                        doc_name=document_name,
                    )
                    count += 1

            if entry.sub_sub_bab and entry.sub_bab:
                sub_sub_bab_key = (bab_name, entry.sub_bab, entry.sub_sub_bab)
                if sub_sub_bab_key not in seen_sub_sub_babs:
                    seen_sub_sub_babs.add(sub_sub_bab_key)
                    session.run(
                        """
                        MERGE (ssb:SubSubBab {name: $sub_sub_bab_name, sub_bab: $sub_bab_name, bab: $bab_name, document: $doc_name})
                        WITH ssb
                        MATCH (sb:SubBab {name: $sub_bab_name, bab: $bab_name, document: $doc_name})
                        MERGE (sb)-[:hasSubSubBab]->(ssb)
                        """,
                        sub_sub_bab_name=entry.sub_sub_bab,
                        sub_bab_name=entry.sub_bab,
                        bab_name=bab_name,
                        doc_name=document_name,
                    )
                    count += 1

    return count


def insert_konsep(
    driver,
    data: dict,
    document_name: str = "Unknown",
    kelas: str | None = None,
    subject_name: str | None = None,
    subject_phase: str | None = None,
):
    """Insert konsep into Neo4j following the new KG ontology.

    Creates:
      (:MataPelajaran)-[:hasDocument]->(:Document {kelas})-[:hasBab]->(:Bab)
        -[:hasSubBab]->(:SubBab)-[:hasSubSubBab]->(:SubSubBab)-[:hasKonsep]->(:Konsep)
        -[:hasSubKonsep]->(:SubKonsep)

    When sub_bab is None, Konsep is linked directly to Bab (no SubBab node).
    When sub_sub_bab is None, Konsep is linked to SubBab (no SubSubBab node).
    Konsep are shared across documents (MERGE on name) to enable cross-doc linking.
    """
    with driver.session() as session:
        # Create the Document node
        session.run(
            """
            MERGE (d:Document {name: $doc_name})
            SET d.uploaded_at = timestamp(), d.kelas = $kelas
            """,
            doc_name=document_name,
            kelas=kelas or "",
        )

        # Create MataPelajaran node and link to Document if provided
        if subject_name:
            session.run(
                """
                MERGE (mp:MataPelajaran {name: $subject_name})
                SET mp.phase = $phase
                WITH mp
                MATCH (d:Document {name: $doc_name})
                MERGE (mp)-[:hasDocument]->(d)
                """,
                subject_name=subject_name,
                phase=subject_phase or "",
                doc_name=document_name,
            )

        # Group konsep by (bab, sub_bab, sub_sub_bab)
        # Skip concepts without bab assignment (ToC is source of truth)
        # Skip concepts with non-ToC bab names (e.g., LLM-detected "Meiosis")
        bab_map: dict[str, dict[str | None, dict[str | None, list[dict]]]] = {}
        skipped_no_bab = 0
        skipped_invalid_bab = 0
        for konsep in data.get("konsep", []):
            bab_name = konsep.get("bab")
            if not bab_name:
                logger.warning(
                    "Konsep '%s' has no bab assignment, skipping",
                    konsep.get("name", "<unnamed>"),
                )
                skipped_no_bab += 1
                continue
            # Validate bab_name follows ToC format (starts with "Bab " or "BAB ")
            if not (bab_name.startswith("Bab ") or bab_name.startswith("BAB ")):
                logger.warning(
                    "Konsep '%s' has non-ToC bab name '%s', skipping",
                    konsep.get("name", "<unnamed>"),
                    bab_name,
                )
                skipped_invalid_bab += 1
                continue
            sub_bab_name: str | None = konsep.get("sub_bab") or None
            sub_sub_bab_name: str | None = konsep.get("sub_sub_bab") or None
            bab_map.setdefault(bab_name, {}).setdefault(sub_bab_name, {}).setdefault(
                sub_sub_bab_name, []
            ).append(konsep)

        if skipped_no_bab > 0:
            logger.info(
                "Skipped %d konsep without bab assignment (ToC is source of truth)",
                skipped_no_bab,
            )
        if skipped_invalid_bab > 0:
            logger.info(
                "Skipped %d konsep with non-ToC bab names (LLM-detected names rejected)",
                skipped_invalid_bab,
            )

        for bab_name, sub_bab_dict in bab_map.items():
            # Only link to existing Bab nodes (created by seed_from_daftar_isi)
            # Do NOT create new Bab nodes here - ToC is the single source of truth
            result = session.run(
                """
                MATCH (b:Bab {name: $bab_name, document: $doc_name})
                RETURN b.name as found
                """,
                bab_name=bab_name,
                doc_name=document_name,
            )
            record = result.single()
            if not record:
                konsep_count = sum(
                    len(kl)
                    for sb_dict in sub_bab_dict.values()
                    for kl in sb_dict.values()
                )
                logger.warning(
                    "Bab '%s' not found in graph (not in ToC?), skipping %d konsep",
                    bab_name,
                    konsep_count,
                )
                continue

            for sub_bab_name, sub_sub_bab_dict in sub_bab_dict.items():
                if sub_bab_name:
                    # Create SubBab node linked to Bab
                    session.run(
                        """
                        MERGE (sb:SubBab {name: $sub_bab_name, bab: $bab_name, document: $doc_name})
                        WITH sb
                        MATCH (b:Bab {name: $bab_name, document: $doc_name})
                        MERGE (b)-[:hasSubBab]->(sb)
                        """,
                        sub_bab_name=sub_bab_name,
                        bab_name=bab_name,
                        doc_name=document_name,
                    )

                for sub_sub_bab_name, konsep_list in sub_sub_bab_dict.items():
                    if sub_bab_name and sub_sub_bab_name:
                        # Create SubSubBab node linked to SubBab
                        session.run(
                            """
                            MERGE (ssb:SubSubBab {name: $sub_sub_bab_name, sub_bab: $sub_bab_name, bab: $bab_name, document: $doc_name})
                            WITH ssb
                            MATCH (sb:SubBab {name: $sub_bab_name, bab: $bab_name, document: $doc_name})
                            MERGE (sb)-[:hasSubSubBab]->(ssb)
                            """,
                            sub_sub_bab_name=sub_sub_bab_name,
                            sub_bab_name=sub_bab_name,
                            bab_name=bab_name,
                            doc_name=document_name,
                        )

                    for konsep in konsep_list:
                        # MERGE Konsep (shared across documents)
                        session.run(
                            """
                            MERGE (k:Konsep {name: $name})
                            SET k.description = $desc,
                                k.formula = $formula,
                                k.variables = $variables,
                                k.kondisi = $kondisi
                            """,
                            name=konsep["name"],
                            desc=konsep.get("description", ""),
                            formula=konsep.get("formula", []),
                            variables=konsep.get("variables", []),
                            kondisi=konsep.get("kondisi", []),
                        )

                        # Link to the most specific structural node available
                        if sub_bab_name and sub_sub_bab_name:
                            # Link SubSubBab -> Konsep
                            session.run(
                                """
                                MATCH (ssb:SubSubBab {name: $sub_sub_bab_name, sub_bab: $sub_bab_name, bab: $bab_name, document: $doc_name})
                                MATCH (k:Konsep {name: $konsep_name})
                                MERGE (ssb)-[:hasKonsep]->(k)
                                """,
                                sub_sub_bab_name=sub_sub_bab_name,
                                sub_bab_name=sub_bab_name,
                                bab_name=bab_name,
                                doc_name=document_name,
                                konsep_name=konsep["name"],
                            )
                        elif sub_bab_name:
                            # Link SubBab -> Konsep
                            session.run(
                                """
                                MATCH (sb:SubBab {name: $sub_bab_name, bab: $bab_name, document: $doc_name})
                                MATCH (k:Konsep {name: $konsep_name})
                                MERGE (sb)-[:hasKonsep]->(k)
                                """,
                                sub_bab_name=sub_bab_name,
                                bab_name=bab_name,
                                doc_name=document_name,
                                konsep_name=konsep["name"],
                            )
                        else:
                            # Fallback: link Bab -> Konsep directly (no SubBab)
                            session.run(
                                """
                                MATCH (b:Bab {name: $bab_name, document: $doc_name})
                                MATCH (k:Konsep {name: $konsep_name})
                                MERGE (b)-[:hasKonsep]->(k)
                                """,
                                bab_name=bab_name,
                                doc_name=document_name,
                                konsep_name=konsep["name"],
                            )

                        # Handle sub-konsep
                        for sub in konsep.get("sub_konsep", []):
                            session.run(
                                """
                                MERGE (sk:SubKonsep {name: $name})
                                SET sk.description = $desc,
                                    sk.formula = $formula,
                                    sk.variables = $variables,
                                    sk.kondisi = $kondisi
                                WITH sk
                                MATCH (k:Konsep {name: $konsep_name})
                                MERGE (k)-[:hasSubKonsep]->(sk)
                                """,
                                name=sub["name"],
                                desc=sub.get("description", ""),
                                formula=sub.get("formula", []),
                                variables=sub.get("variables", []),
                                kondisi=sub.get("kondisi", []),
                                konsep_name=konsep["name"],
                            )

    # -------------------------------------------------------------------------
    # PASS 2: Create inline relations from per-Bab extraction
    # -------------------------------------------------------------------------
    from src.schemas import normalize_relation_type

    with driver.session() as session:
        for konsep in data.get("konsep", []):
            konsep_name = konsep.get("name")
            if not konsep_name:
                continue

            for rel in konsep.get("relations", []):
                rel_type = rel.get("type", "")
                target_name = rel.get("target", "")
                description = rel.get("description", "")

                # Normalize relation type to ontology-defined types
                normalized_type = normalize_relation_type(rel_type)
                if normalized_type is None:
                    logger.debug(
                        "Skipping unmapped relation type '%s' from '%s'",
                        rel_type,
                        konsep_name,
                    )
                    continue

                if not target_name:
                    continue

                # Use normalized type for Cypher
                safe_rel_type = normalized_type

                # Create relation (MERGE target to handle forward references)
                try:
                    session.run(
                        f"""
                        MATCH (a:Konsep {{name: $source}})
                        MERGE (b:Konsep {{name: $target}})
                        MERGE (a)-[r:{safe_rel_type}]->(b)
                        SET r.description = $desc
                        """,
                        source=konsep_name,
                        target=target_name,
                        desc=description,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to create relation %s -> %s: %s",
                        konsep_name,
                        target_name,
                        e,
                    )

    # -------------------------------------------------------------------------
    # PASS 3: Create Variable nodes and USES_VARIABLE edges
    # -------------------------------------------------------------------------
    with driver.session() as session:
        for konsep in data.get("konsep", []):
            all_nodes = [konsep] + konsep.get("sub_konsep", [])
            for node in all_nodes:
                node_name = node.get("name")
                if not node_name:
                    continue
                label = "SubKonsep" if node is not konsep else "Konsep"
                for var_str in node.get("variables", []):
                    _insert_variable(session, node_name, label, var_str)

    # PASS 4: Create SHARES_VARIABLE edges between Konsep sharing a variable
    with driver.session() as session:
        session.run(
            """
            MATCH (a:Konsep)-[:USES_VARIABLE]->(v:Variable)<-[:USES_VARIABLE]-(b:Konsep)
            WHERE a <> b AND a.name < b.name
            MERGE (a)-[:SHARES_VARIABLE {variable: v.name}]->(b)
            """
        )


def _insert_variable(session, node_name: str, label: str, var_str: str) -> None:
    """Parse a variable string and create Variable node + USES_VARIABLE edge.

    Accepts formats like:
      "F (gaya, N)"    → name="F", label="gaya", unit="N"
      "m (massa, kg)"  → name="m", label="massa", unit="kg"
      "v"              → name="v", label="", unit=""
    """
    import re

    var_str = var_str.strip()
    if not var_str:
        return

    match = re.match(r"^([^\s(]+)\s*\(([^,)]+)(?:,\s*([^)]+))?\)", var_str)
    if match:
        var_name = match.group(1).strip()
        var_label = match.group(2).strip()
        var_unit = match.group(3).strip() if match.group(3) else ""
    else:
        var_name = var_str
        var_label = ""
        var_unit = ""

    try:
        session.run(
            f"""
            MERGE (v:Variable {{name: $var_name}})
            SET v.label = $var_label, v.unit = $var_unit
            WITH v
            MATCH (n:{label} {{name: $node_name}})
            MERGE (n)-[:USES_VARIABLE]->(v)
            """,
            var_name=var_name,
            var_label=var_label,
            var_unit=var_unit,
            node_name=node_name,
        )
    except Exception as e:
        logger.warning(
            "Failed to insert variable '%s' for '%s': %s", var_str, node_name, e
        )


# Backward-compat wrapper
def insert_topics(
    driver,
    data: dict,
    document_name: str = "Unknown",
    subject_name: str | None = None,
    subject_phase: str | None = None,
):
    """Legacy wrapper for insert_konsep. Converts old format if needed."""
    # Handle old format {"topics": [...]} -> {"konsep": [...]}
    if "topics" in data and "konsep" not in data:
        konsep = []
        for t in data.get("topics", []):
            sub_konsep = [
                {
                    "name": s["name"],
                    "description": s.get("description", ""),
                }
                for s in t.get("sub_topics", [])
            ]
            konsep.append(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "bab": None,
                    "sub_konsep": sub_konsep,
                }
            )
        data = {"konsep": konsep}

    insert_konsep(
        driver,
        data,
        document_name=document_name,
        subject_name=subject_name,
        subject_phase=subject_phase,
    )


def get_all_nodes(driver) -> list:
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN n.name AS name, labels(n) AS labels")
        return [dict(r) for r in result]


def get_all_documents(driver) -> list:
    """Get all documents in the knowledge graph."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:hasBab]->(:Bab)-[:hasKonsep]->(k:Konsep)
            RETURN d.name AS name, d.kelas AS kelas, d.uploaded_at AS uploaded_at,
                   count(DISTINCT k) AS topic_count
            ORDER BY d.uploaded_at DESC
            """
        )
        return [dict(r) for r in result]


def get_document_topics(driver, document_name: str) -> list:
    """Get all konsep for a specific document."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:Document {name: $doc_name})-[:hasBab]->(b:Bab)-[:hasKonsep]->(k:Konsep)
            OPTIONAL MATCH (k)-[:hasSubKonsep]->(sk:SubKonsep)
            RETURN k.name AS topic, k.description AS topic_desc,
                   b.name AS bab,
                   collect({name: sk.name, description: sk.description}) AS subtopics
            """,
            doc_name=document_name,
        )
        return [dict(r) for r in result]


def get_graph_stats(driver) -> dict:
    """Get statistics about the knowledge graph."""
    with driver.session() as session:
        result = session.run(
            """
            OPTIONAL MATCH (mp:MataPelajaran) WITH count(mp) AS subjects
            OPTIONAL MATCH (d:Document) WITH subjects, count(d) AS documents
            OPTIONAL MATCH (b:Bab) WITH subjects, documents, count(b) AS babs
            OPTIONAL MATCH (sb:SubBab) WITH subjects, documents, babs, count(sb) AS sub_babs
            OPTIONAL MATCH (k:Konsep) WITH subjects, documents, babs, sub_babs, count(k) AS konsep
            OPTIONAL MATCH (sk:SubKonsep) WITH subjects, documents, babs, sub_babs, konsep, count(sk) AS sub_konsep
            OPTIONAL MATCH ()-[r:SIMILAR_TO]->() WITH subjects, documents, babs, sub_babs, konsep, sub_konsep, count(r) AS similar_rels
            OPTIONAL MATCH ()-[r2:isPrerequisiteOf]->() WITH subjects, documents, babs, sub_babs, konsep, sub_konsep, similar_rels, count(r2) AS prereq_rels
            OPTIONAL MATCH ()-[r3:supports]->() WITH subjects, documents, babs, sub_babs, konsep, sub_konsep, similar_rels, prereq_rels, count(r3) AS supports_rels
            OPTIONAL MATCH ()-[r4:analogousTo]->() WITH subjects, documents, babs, sub_babs, konsep, sub_konsep, similar_rels, prereq_rels, supports_rels, count(r4) AS analogous_rels
            OPTIONAL MATCH ()-[r5:BAGIAN_DARI|MENYEBABKAN|BERGANTUNG_PADA|MENDEFINISIKAN|MEMPENGARUHI|BERINTERAKSI_DENGAN|DIFORMULASIKAN_SEBAGAI]->()
            RETURN subjects, documents, babs, sub_babs, konsep, sub_konsep, similar_rels,
                   prereq_rels, supports_rels, analogous_rels, count(r5) AS inline_rels
            """
        )
        row = result.single()
        if row:
            return {
                "subjects": row["subjects"],
                "documents": row["documents"],
                "babs": row["babs"],
                "sub_babs": row["sub_babs"],
                "topics": row["konsep"],  # keep "topics" key for backward compat
                "subtopics": row[
                    "sub_konsep"
                ],  # keep "subtopics" key for backward compat
                "konsep": row["konsep"],
                "sub_konsep": row["sub_konsep"],
                "similar_rels": row["similar_rels"],
                "prereq_rels": row["prereq_rels"],
                "supports_rels": row["supports_rels"],
                "analogous_rels": row["analogous_rels"],
                "inline_rels": row["inline_rels"],
            }
        return {
            "subjects": 0,
            "documents": 0,
            "babs": 0,
            "sub_babs": 0,
            "topics": 0,
            "subtopics": 0,
            "konsep": 0,
            "sub_konsep": 0,
            "similar_rels": 0,
            "prereq_rels": 0,
            "supports_rels": 0,
            "analogous_rels": 0,
            "inline_rels": 0,
        }


def get_all_topics_with_subtopics(driver) -> list:
    """Get all konsep with their sub-konsep across all documents."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (k:Konsep)
            OPTIONAL MATCH (k)-[:hasSubKonsep]->(sk:SubKonsep)
            OPTIONAL MATCH (b:Bab)-[:hasKonsep]->(k)
            OPTIONAL MATCH (d:Document)-[:hasBab]->(b)
            RETURN k.name AS name, k.description AS description,
                   collect(DISTINCT {name: sk.name, description: sk.description}) AS subtopics,
                   collect(DISTINCT d.name) AS documents
            ORDER BY k.name
            """
        )
        return [dict(r) for r in result]


def get_similar_relationships(driver) -> list:
    """Get all SIMILAR_TO relationships."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a)-[r:SIMILAR_TO]->(b)
            RETURN a.name AS source, b.name AS target, r.score AS score
            ORDER BY r.score DESC
            """
        )
        return [dict(r) for r in result]


def get_typed_relationships(driver) -> list:
    """Get all typed concept relationships (both completion and inline extraction)."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE type(r) IN ['isPrerequisiteOf', 'supports', 'analogousTo',
                              'BAGIAN_DARI', 'MENYEBABKAN', 'BERGANTUNG_PADA',
                              'MENDEFINISIKAN', 'MEMPENGARUHI', 'BERINTERAKSI_DENGAN',
                              'DIFORMULASIKAN_SEBAGAI']
               OR type(r) STARTS WITH 'LINTAS_BUKU'
            RETURN a.name AS source, b.name AS target,
                   type(r) AS rel_type, r.confidence AS confidence, r.description AS description
            ORDER BY rel_type, source
            """
        )
        return [dict(r) for r in result]


def create_typed_relationship(
    driver,
    source: str,
    target: str,
    rel_type: str,
    confidence: float | None = None,
) -> None:
    """Create a typed relationship between two Konsep nodes."""
    valid_types = {"isPrerequisiteOf", "supports", "analogousTo"}
    if rel_type not in valid_types:
        raise ValueError(f"rel_type must be one of {valid_types}")

    with driver.session() as session:
        session.run(
            f"""
            MATCH (a:Konsep {{name: $src}})
            MATCH (b:Konsep {{name: $tgt}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.confidence = $confidence
            """,
            src=source,
            tgt=target,
            confidence=confidence,
        )


def get_nodes_with_descriptions(
    driver,
    document_name: str | None = None,
    include_cross_doc: bool = False,
) -> list[dict]:
    """Get Konsep nodes with their names and descriptions for embedding.

    Args:
        driver: Neo4j driver
        document_name: If provided, filter to this document only
        include_cross_doc: If True with document_name, also include nodes from
                          other documents

    Returns list of dicts with 'name', 'description', 'labels', and 'document'.
    Falls back to name as description if description is empty.
    """
    with driver.session() as session:
        if document_name and not include_cross_doc:
            result = session.run(
                """
                MATCH (d:Document {name: $doc_name})-[:hasBab]->(b:Bab)-[:hasKonsep]->(k:Konsep)
                RETURN k.name AS name, k.description AS description,
                       labels(k) AS labels, d.name AS doc_name,
                       coalesce(k.formula, []) AS formula,
                       coalesce(k.variables, []) AS variables,
                       coalesce(k.kondisi, []) AS kondisi
                UNION
                MATCH (d:Document {name: $doc_name})-[:hasBab]->(:Bab)-[:hasKonsep]->(:Konsep)
                      -[:hasSubKonsep]->(sk:SubKonsep)
                RETURN sk.name AS name, sk.description AS description,
                       labels(sk) AS labels, d.name AS doc_name,
                       coalesce(sk.formula, []) AS formula,
                       coalesce(sk.variables, []) AS variables,
                       coalesce(sk.kondisi, []) AS kondisi
                ORDER BY name
                """,
                doc_name=document_name,
            )
        elif document_name and include_cross_doc:
            result = session.run(
                """
                MATCH (n)
                WHERE n:Konsep OR n:SubKonsep
                RETURN n.name AS name, n.description AS description,
                       labels(n) AS labels, NULL AS doc_name,
                       coalesce(n.formula, []) AS formula,
                       coalesce(n.variables, []) AS variables,
                       coalesce(n.kondisi, []) AS kondisi
                ORDER BY name
                """,
                doc_name=document_name,
            )
        else:
            result = session.run(
                """
                MATCH (n)
                WHERE n:Konsep OR n:SubKonsep
                OPTIONAL MATCH (d:Document)-[:hasBab]->(:Bab)-[:hasKonsep]->(n)
                OPTIONAL MATCH (d2:Document)-[:hasBab]->(:Bab)-[:hasKonsep]->(:Konsep)
                       -[:hasSubKonsep]->(n)
                WITH n, COALESCE(d.name, d2.name) AS doc_name
                RETURN n.name AS name, n.description AS description,
                       labels(n) AS labels, doc_name,
                       coalesce(n.formula, []) AS formula,
                       coalesce(n.variables, []) AS variables,
                       coalesce(n.kondisi, []) AS kondisi
                ORDER BY n.name
                """
            )

        nodes = []
        seen_names = set()
        for r in result:
            name = r["name"]
            if name in seen_names:
                continue
            seen_names.add(name)
            desc = r["description"] or name
            nodes.append(
                {
                    "name": name,
                    "description": desc,
                    "labels": r["labels"],
                    "document": r.get("doc_name"),
                    "formula": list(r["formula"]) if r.get("formula") else [],
                    "variables": list(r["variables"]) if r.get("variables") else [],
                    "kondisi": list(r["kondisi"]) if r.get("kondisi") else [],
                }
            )
        return nodes


def get_concept_context(
    driver,
    names: list[str],
    max_edges: int = 5,
) -> dict[str, dict]:
    """Fetch graph context for a batch of Konsep/SubKonsep nodes.

    Returns {name: {"bab": str|None, "out_edges": [{rel_type, target, confidence}]}}.
    Out-edges are sorted by confidence desc and capped at max_edges. The Bab walk
    follows structural edges (hasSubBab/hasSubSubBab/hasKonsep/hasSubKonsep) so
    SubKonsep nodes also resolve to their parent Bab.
    """
    if not names:
        return {}
    contexts: dict[str, dict] = {}
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n) WHERE (n:Konsep OR n:SubKonsep) AND n.name IN $names
            OPTIONAL MATCH (n)-[r:isPrerequisiteOf|supports|analogousTo]->(t)
            WITH n, collect(DISTINCT {
                    rel_type: type(r),
                    target: t.name,
                    confidence: r.confidence
                 }) AS edges
            OPTIONAL MATCH (b:Bab)-[:hasSubBab|hasSubSubBab|hasKonsep|hasSubKonsep*1..4]->(n)
            WITH n, edges, head(collect(DISTINCT b.name)) AS bab
            RETURN n.name AS name, bab,
                   [e IN edges WHERE e.target IS NOT NULL] AS out_edges
            """,
            names=names,
        )
        for r in result:
            edges = sorted(
                r["out_edges"],
                key=lambda e: (e.get("confidence") or 0.0),
                reverse=True,
            )[:max_edges]
            contexts[r["name"]] = {
                "bab": r["bab"],
                "out_edges": edges,
            }
    for name in names:
        contexts.setdefault(name, {"bab": None, "out_edges": []})
    return contexts


def get_graph_quality_metrics(driver) -> dict:
    """Calculate graph quality metrics: ADC and Modularity.

    Uses Konsep and SubKonsep nodes and their relationships.
    """
    G = nx.Graph()

    with driver.session() as session:
        nodes = session.run(
            """
            MATCH (n) WHERE n:Konsep OR n:SubKonsep
            RETURN elementId(n) AS id, n.name AS name
            """
        )
        for r in nodes:
            G.add_node(r["id"])

        edges = session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE (a:Konsep OR a:SubKonsep) AND (b:Konsep OR b:SubKonsep)
            RETURN elementId(a) AS src, elementId(b) AS tgt
            """
        )
        for r in edges:
            G.add_edge(r["src"], r["tgt"])

    if G.number_of_nodes() == 0:
        return {"adc": 0.0, "modularity": 0.0, "density": 0.0, "components": 0}

    centrality = nx.degree_centrality(G)
    adc = sum(centrality.values()) / len(centrality)

    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        modularity = (
            nx.community.modularity(G, communities) if len(communities) > 1 else 0.0
        )
    except Exception:
        modularity = 0.0

    return {
        "adc": round(adc, 3),
        "modularity": round(modularity, 3),
        "density": round(nx.density(G), 3),
        "components": nx.number_connected_components(G),
    }


# --- Vector Index Functions for ANN Search ---


def get_embedding_dimensions(model: str) -> int:
    """Get embedding dimensions for a given model."""
    # Mapping of model patterns to their dimensions
    dimension_map = {
        "gemini-embedding": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "multilingual-e5-small": 384,
        "bge-small": 384,
        "bge-base": 768,
        "bge-large": 1024,
        "Qwen3-Embedding": 4096,
    }
    for pattern, dim in dimension_map.items():
        if pattern.lower() in model.lower():
            return dim
    # Default to 768 (common for many models)
    return 768


def create_vector_index(
    driver,
    index_name: str = "konsep_embedding_idx",
    dimensions: int = 768,
) -> bool:
    """Create (or refresh) vector indexes for Konsep and SubKonsep nodes.

    If an index already exists at a different dimension, it is dropped and
    recreated at the requested dimension. This is the behavior the UI's
    "Create/Refresh Index" button promises.

    Returns True if successful, False otherwise.
    """
    specs = [
        (index_name, "Konsep"),
        ("subkonsep_embedding_idx", "SubKonsep"),
    ]
    try:
        with driver.session() as session:
            for name, label in specs:
                existing = session.run(
                    "SHOW INDEXES YIELD name, options "
                    "WHERE name = $name "
                    "RETURN options.indexConfig.`vector.dimensions` AS dim",
                    name=name,
                ).single()
                if existing is not None and existing["dim"] != dimensions:
                    logger.warning(
                        "Dropping stale %s index (was %d-dim, need %d-dim)",
                        name,
                        existing["dim"],
                        dimensions,
                    )
                    session.run(f"DROP INDEX {name} IF EXISTS")

                session.run(
                    f"""
                    CREATE VECTOR INDEX {name} IF NOT EXISTS
                    FOR (n:{label}) ON n.embedding
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimensions},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                )
        logger.info("Vector indexes ready at %d dimensions", dimensions)
        return True
    except Exception as e:
        logger.error("Failed to create vector index: %s", e)
        return False


def drop_vector_index(
    driver,
    index_name: str = "konsep_embedding_idx",
) -> bool:
    """Drop a vector index."""
    try:
        with driver.session() as session:
            session.run(f"DROP INDEX {index_name} IF EXISTS")
            session.run("DROP INDEX subkonsep_embedding_idx IF EXISTS")
        logger.info("Dropped vector indexes")
        return True
    except Exception as e:
        logger.error("Failed to drop vector index: %s", e)
        return False


def get_nodes_with_current_embedding(
    driver,
    names: list[str],
    model: str,
) -> set[str]:
    """Return the subset of `names` whose Konsep/SubKonsep node already has
    an embedding stored under `model`. Used to skip redundant re-uploads
    of vectors that Neo4j already holds."""
    if not names:
        return set()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n) WHERE (n:Konsep OR n:SubKonsep)
                AND n.name IN $names
                AND n.embedding IS NOT NULL
                AND n.embedding_model = $model
            RETURN n.name AS name
            """,
            names=names,
            model=model,
        )
        return {r["name"] for r in result}


def store_node_embeddings(
    driver,
    nodes: list[dict],
    embeddings: list[list[float]],
    model: str,
) -> int:
    """Store embeddings on Konsep and SubKonsep nodes.

    Args:
        driver: Neo4j driver
        nodes: List of dicts with 'name' and 'labels' keys
        embeddings: Corresponding embedding vectors
        model: Model name used for embedding (stored for tracking)

    Returns number of nodes updated.
    """
    updated = 0
    with driver.session() as session:
        for node, emb in zip(nodes, embeddings):
            label_filter = (
                "Konsep" if "Konsep" in node.get("labels", []) else "SubKonsep"
            )
            session.run(
                f"""
                MATCH (n:{label_filter} {{name: $name}})
                SET n.embedding = $embedding, n.embedding_model = $model
                """,
                name=node["name"],
                embedding=emb,
                model=model,
            )
            updated += 1
    logger.info("Stored embeddings on %d nodes (model: %s)", updated, model)
    return updated


def query_similar_nodes(
    driver,
    query_embedding: list[float],
    k: int = 10,
    threshold: float = 0.8,
    exclude_name: str | None = None,
    index_name: str = "konsep_embedding_idx",
) -> list[dict]:
    """Query for similar nodes using vector index (ANN search).

    Args:
        driver: Neo4j driver
        query_embedding: The embedding vector to search with
        k: Number of nearest neighbors to return
        threshold: Minimum similarity score
        exclude_name: Optional node name to exclude from results
        index_name: Name of the vector index to use

    Returns list of dicts with 'name', 'description', 'score', 'labels'.
    """
    with driver.session() as session:
        # Query Konsep index
        result = session.run(
            f"""
            CALL db.index.vector.queryNodes('{index_name}', $k, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            AND ($exclude_name IS NULL OR node.name <> $exclude_name)
            RETURN node.name AS name, node.description AS description,
                   score, labels(node) AS labels
            ORDER BY score DESC
            """,
            embedding=query_embedding,
            k=k,
            threshold=threshold,
            exclude_name=exclude_name,
        )
        konsep_results = [dict(r) for r in result]

        # Query SubKonsep index
        result2 = session.run(
            """
            CALL db.index.vector.queryNodes('subkonsep_embedding_idx', $k, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            AND ($exclude_name IS NULL OR node.name <> $exclude_name)
            RETURN node.name AS name, node.description AS description,
                   score, labels(node) AS labels
            ORDER BY score DESC
            """,
            embedding=query_embedding,
            k=k,
            threshold=threshold,
            exclude_name=exclude_name,
        )
        subkonsep_results = [dict(r) for r in result2]

    # Merge and deduplicate
    all_results = konsep_results + subkonsep_results
    seen = set()
    unique = []
    for r in all_results:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)
    return sorted(unique, key=lambda x: x["score"], reverse=True)[:k]


def get_index_info(driver, index_name: str = "konsep_embedding_idx") -> dict | None:
    """Get information about a vector index."""
    try:
        with driver.session() as session:
            result = session.run(
                """
                SHOW INDEXES YIELD name, type, state, indexProvider
                WHERE name = $index_name OR name = 'subkonsep_embedding_idx'
                RETURN name, type, state, indexProvider
                """,
                index_name=index_name,
            )
            indexes = [dict(r) for r in result]
            if indexes:
                return {
                    "exists": True,
                    "indexes": indexes,
                    "state": indexes[0].get("state", "UNKNOWN"),
                }
            return {"exists": False}
    except Exception as e:
        logger.error("Failed to get index info: %s", e)
        return {"exists": False, "error": str(e)}


def count_nodes_with_embeddings(driver) -> dict:
    """Count nodes that have embeddings stored."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n) WHERE n:Konsep OR n:SubKonsep
            WITH count(n) AS total,
                 sum(CASE WHEN n.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_embedding,
                 collect(DISTINCT n.embedding_model) AS models
            RETURN total, with_embedding, models
            """
        )
        r = result.single()
        if r:
            return {
                "total": r["total"],
                "with_embedding": r["with_embedding"],
                "models": [m for m in r["models"] if m],
            }
        return {"total": 0, "with_embedding": 0, "models": []}
