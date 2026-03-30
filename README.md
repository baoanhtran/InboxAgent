# Gmail Executive Assistant — Agent IA Multi-Agents

Un assistant email intelligent basé sur une architecture **multi-agents LangGraph** qui lit votre boîte Gmail, analyse le contexte, rédige des réponses professionnelles et les envoie après validation humaine.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture](#2-architecture)
3. [Les agents spécialistes](#3-les-agents-spécialistes)
4. [Flux d'exécution détaillé](#4-flux-dexécution-détaillé)
5. [Structure du projet](#5-structure-du-projet)
6. [Installation et configuration](#6-installation-et-configuration)
7. [Configuration des modèles IA](#7-configuration-des-modèles-ia)
8. [Lancer l'application](#8-lancer-lapplication)
9. [Interaction avec le terminal](#9-interaction-avec-le-terminal)
10. [Concepts clés](#10-concepts-clés)

---

## 1. Vue d'ensemble

Ce projet implémente un **assistant exécutif pour Gmail** entièrement piloté par des agents IA. Son rôle est de :

- Surveiller la boîte de réception et identifier l'email non lu le plus récent
- Comprendre le contexte (historique de conversation, profil de l'expéditeur)
- Rédiger une réponse professionnelle adaptée
- Soumettre la réponse à une révision humaine avant envoi

Le principe fondamental est le **Human-in-the-Loop** : l'agent ne peut ni lire ni envoyer d'email sans approbation explicite de l'utilisateur à chaque étape.

Le système est **model-agnostic** : chaque agent peut utiliser un modèle différent (OpenAI, Anthropic, Google, Mistral, Ollama) configuré via variables d'environnement.

---

## 2. Architecture

Le système repose sur un **Coordinateur méta-agent** qui orchestre une équipe de 6 agents spécialistes. Chaque agent est un sous-graphe LangGraph indépendant.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              GRAPHE PRINCIPAL                           │
│                                                                          │
│   coordinator_agent (décide le prochain spécialiste)                    │
│      │ next_action                                                       │
│      ├─ inbox_scanner ──┬─ no_unread_emails → END                       │
│      │                  └─ email_content + email_attachments            │
│      ├─ attachment_analyzer (si pièces jointes et pas de résumé)         │
│      ├─ thread_researcher (si nécessaire)                                │
│      ├─ sender_profiler (si nécessaire)                                  │
│      ├─ composer (sans accès Gmail)                                      │
│      └─ reviewer (sans accès Gmail, optionnel)                           │
│                                                                          │
│   FINISH → human_review_node (pause utilisateur)                         │
│      ├─ approve/edit → mcp_executor_node → END                           │
│      └─ reject       → coordinator_agent                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

### Principe de routing

Le coordinateur analyse l'état global après chaque spécialiste et décide de la prochaine action en sortant un JSON :

```json
{ "next": "composer", "reason": "Email content fetched, ready to draft reply" }
```

| Chemin                                                                               | Cas d'usage               |
| ------------------------------------------------------------------------------------ | ------------------------- |
| `inbox_scanner → END`                                                                | Aucun email non lu        |
| `inbox_scanner → composer → FINISH`                                                  | Email simple              |
| `inbox_scanner → attachment_analyzer → composer → FINISH`                            | Email avec pièces jointes |
| `inbox_scanner → thread_researcher → sender_profiler → composer → reviewer → FINISH` | Email complexe            |

Un garde-fou limite les itérations à **10 maximum** pour éviter les boucles infinies.

---

## 3. Les agents spécialistes

### `inbox_scanner` — Scanner de boîte de réception

**Rôle :** Trouver le dernier email non lu et en extraire le contenu.

**Comportement :**

1. Le LLM dispose uniquement de `list_emails` — il ne peut pas appeler `get_email` directement
2. Après approbation, le code extrait le `messageId` du résultat de `list_emails` via 4 patterns regex (robustesse face aux formats variables du serveur MCP)
3. `get_email` est appelé **en code** dans `scanner_finalize_node` — jamais via LLM, pour garantir l'exactitude du contenu
4. Après `get_email`, les métadonnées de pièces jointes sont extraites via `extract_attachments_from_get_result()` et stockées dans `email_attachments` au format `[{resourceUri, filename, mimeType}]`
5. Si aucun email non lu : `status = "no_unread_emails"` → le graphe route vers `END` sans passer par `human_review_node`

---

### `attachment_analyzer` — Analyseur de pièces jointes

**Rôle :** Lire et résumer les pièces jointes de l'email pour enrichir la réponse.

**Comportement :**

- Le coordinateur le déclenche si `ATTACHMENTS > 0` et `attachment_summary` est vide
- Pour chaque fichier, l'agent lit la ressource binaire via `read_mcp_resource(resourceUri)`
- Selon le type MIME, il applique une stratégie adaptée :
    - `image/*` → OpenAI Chat Completions avec `image_url` en base64
    - `application/pdf` → OpenAI Responses API avec `input_file` base64
    - `text/*` → décodage base64 puis synthèse via OpenAI Chat Completions
- Le résumé consolidé est écrit dans `attachment_summary`

---

### `thread_researcher` — Chercheur de fil de discussion

**Rôle :** Lire l'historique complet de la conversation pour comprendre le contexte.

**Comportement :**

- Utilise `list_emails` et `get_email` pour retrouver les messages précédents du même fil
- Produit un résumé : historique, décisions passées, engagements en cours, ton général
- Résultat stocké dans `thread_context`

---

### `sender_profiler` — Profileur de l'expéditeur

**Rôle :** Analyser les emails passés de l'expéditeur pour adapter le ton de la réponse.

**Comportement :**

- Recherche les emails récents puis identifie ceux de l'expéditeur cible
- Construit un profil : ancienneté de la relation, style de communication, sujets récurrents, niveau de formalité recommandé
- Résultat stocké dans `sender_profile`

---

### `composer` — Rédacteur de réponse

**Rôle :** Rédiger une réponse professionnelle en exploitant tout le contexte accumulé.

**Comportement :**

- N'a **aucun accès à Gmail** (`use_tools=False`) — travaille uniquement avec les données en mémoire d'état
- Intègre : `email_content`, `attachment_summary`, `thread_context`, `sender_profile`, et les `review_notes` si c'est une re-rédaction
- Produit uniquement le corps de l'email, sans commentaire ni introduction
- Résultat stocké dans `draft_reply`

---

### `reviewer` — Réviseur de brouillon

**Rôle :** Évaluer la qualité du brouillon avant soumission à l'humain.

**Comportement :**

- N'a **aucun accès à Gmail** (`use_tools=False`) — travaille uniquement avec l'état
- Évalue selon : exhaustivité, ton, clarté, exactitude, représentation appropriée
- Retourne soit `APPROVE` (avec justification), soit `REVISE` (avec liste de points à corriger)
- En cas de `REVISE`, le coordinateur renvoie le composer pour une nouvelle version
- Résultat stocké dans `review_notes`

---

### `human_review_node` — Validation humaine

**Rôle :** Soumettre le brouillon final à l'approbation de l'utilisateur.

**Comportement :**

- Pause l'exécution du graphe via `interrupt()` (mécanisme LangGraph)
- Attend une réponse parmi trois options :
    - `approve` → le brouillon est envoyé tel quel
    - `edit` → l'utilisateur fournit un texte révisé
    - `reject` → renvoi au coordinator pour re-rédaction complète

---

### `mcp_executor_node` — Exécuteur d'envoi

**Rôle :** Envoyer la réponse finale et marquer l'email original comme lu.

**Comportement :**

- Convertit le texte du brouillon en HTML minimal (`<p>`, `<br>`)
- Appelle `reply_to_email` avec l'identifiant du message original — le serveur MCP gère le fil de discussion (sujet `Re:`) automatiquement
- Appelle `mark_email_as_read` pour marquer l'email traité
- Ne nécessite pas d'approbation supplémentaire (déjà validé par `human_review_node`)

---

## 4. Flux d'exécution détaillé

```
python main.py
      │
      ▼
coordinator_agent ──────────────────────────────────────────────┐
      │                                                          │
      │ next_action = "inbox_scanner"                           │
      ▼                                                          │
inbox_scanner_agent                                             │
  ├─ [PAUSE] Tool Approval: list_emails                        │
  │       └─ Utilisateur: y/n                                  │
  ├─ Si aucun email → status="no_unread_emails" → END          │
  ├─ [CODE] get_email(messageId extrait du résultat)            │
  ├─ [CODE] extract_attachments_from_get_result()               │
  └─ → retour au coordinator avec email_content + email_attachments
      │                                                          │
      │ next_action = "attachment_analyzer" (si ATTACHMENTS > 0
      │                                et attachment_summary vide)
      ▼                                                          │
attachment_analyzer_node                                        │
  ├─ Pour chaque fichier: read_mcp_resource(resourceUri)        │
  ├─ image/*         → OpenAI Chat Completions (image_url b64)  │
  ├─ application/pdf → OpenAI Responses API (input_file b64)    │
  ├─ text/*          → decode b64 + Chat Completions            │
  └─ → retour au coordinator avec attachment_summary rempli      │
      │                                                          │
      │ next_action = "thread_researcher" (si nécessaire)       │
      ▼                                                          │
thread_researcher_agent                                         │
  ├─ [PAUSE] Tool Approval: list_emails / get_email            │
  └─ → retour au coordinator avec thread_context rempli         │
      │                                                          │
      │ next_action = "sender_profiler" (si nécessaire)         │
      ▼                                                          │
sender_profiler_agent                                           │
  ├─ [PAUSE] Tool Approval: list_emails / get_email            │
  └─ → retour au coordinator avec sender_profile rempli         │
      │                                                          │
      │ next_action = "composer"                                 │
      ▼                                                          │
composer_agent (aucun accès Gmail)                              │
  └─ → retour au coordinator avec draft_reply rempli            │
      │                                                          │
      │ next_action = "reviewer" (optionnel)                     │
      ▼                                                          │
reviewer_agent (aucun accès Gmail)                              │
  ├─ APPROVE → next_action = "FINISH"                          │
  └─ REVISE  → next_action = "composer" (re-boucle)            │
      │                                                          │
      │ next_action = "FINISH"                                   │
      ▼                                                          │
human_review_node                                               │
  ├─ [PAUSE] Affichage du brouillon                            │
  │       ├─ [a] approve → mcp_executor_node → END             │
  │       ├─ [e] edit    → mcp_executor_node → END             │
  │       └─ [r] reject  ──────────────────────────────────────┘
      │                     (retour au coordinator)
      ▼
mcp_executor_node
  ├─ reply_to_email(messageId, replyBody HTML)
  └─ mark_email_as_read(messageId)
      │
      ▼
     END
```

---

## 5. Structure du projet

```
InboxAgent/
│
├── main.py          # Point d'entrée — boucle interactive terminal
├── graph.py         # Assemblage du graphe LangGraph principal
├── state.py         # Définitions des états TypedDict (un par sous-graphe)
├── llm_config.py    # Factory LLM model-agnostic (multi-providers)
│
├── agents/          # Tous les agents LLM + nœud de validation humaine
│   ├── factory.py          # Factory ReAct générique (agent_node → tool_node → finalize_node)
│   ├── coordinator.py      # Coordinateur méta-agent — décide qui dispatcher
│   ├── inbox_scanner.py    # Scanner boîte de réception
│   ├── attachment_analyzer.py # Analyse multimodale des pièces jointes
│   ├── thread_researcher.py# Chercheur de fil de discussion
│   ├── sender_profiler.py  # Profileur de l'expéditeur
│   ├── composer.py         # Rédacteur de réponse (sans accès Gmail)
│   ├── reviewer.py         # Réviseur de brouillon (sans accès Gmail)
│   └── human_review.py     # Nœud de pause/validation humaine (interrupt)
│
├── gmail/           # Couche d'accès Gmail via MCP
│   ├── client.py    # Client MCP — get_readonly_gmail_tools / get_gmail_tools
│   ├── executor.py  # Nœud d'envoi — reply_to_email + mark_email_as_read
│   └── utils.py     # Parsing MCP : extraction d'ID, décodage MIME RFC 2047, HTML→texte
│
├── .env             # Clés API + config modèles (non versionné)
├── .env.example     # Template de configuration
├── mcp.json         # URL + token du serveur MCP Gmail (non versionné)
├── requirements.txt # Dépendances Python
└── README.md        # Ce fichier
```

### Rôle de chaque module

| Module              | Rôle                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------- |
| `state.py`          | 7 classes `TypedDict` — état global `AgentState` + un état par sous-graphe             |
| `graph.py`          | Routes conditionnelles, garde-fou 10 itérations, compile avec `MemorySaver`            |
| `llm_config.py`     | `get_llm(agent_name)` — lit `<AGENT>_MODEL` depuis `.env`, instancie le bon provider   |
| `agents/factory.py` | Sous-graphe ReAct générique réutilisé par 4 des 6 agents spécialistes                  |
| `gmail/client.py`   | Connexion MCP, expose outils lecture seule et outils complets séparément               |
| `gmail/utils.py`    | 4 patterns d'extraction d'ID, extraction des pièces jointes, décodage MIME, strip HTML |

---

## 6. Installation et configuration

### Prérequis

- Python **3.10** ou supérieur
- Un compte Gmail connecté à un **serveur MCP Gmail** (ex. Cygogn, Composio, ou auto-hébergé)
- Au moins une clé API pour un provider LLM (OpenAI par défaut)

### Étape 1 — Cloner et installer

```bash
git clone <url-du-repo>
cd InboxAgent
pip install -r requirements.txt
```

> Pour utiliser d'autres providers, décommentez les lignes correspondantes dans `requirements.txt` avant d'installer.

### Étape 2 — Créer le fichier `.env`

```bash
cp .env.example .env
```

Voir la [section suivante](#7-configuration-des-modèles-ia) pour la configuration complète.

### Étape 3 — Créer le fichier `mcp.json`

Ce fichier configure la connexion au serveur MCP Gmail. Il doit être placé à la racine du projet.

```json
{
	"mcpServers": {
		"gmail": {
			"url": "https://<adresse-de-votre-serveur-mcp>/mcp",
			"headers": {
				"Authorization": "Bearer <votre-token-jwt>"
			}
		}
	}
}
```

> **Comment obtenir le token ?** Connectez votre compte Gmail au service MCP de votre choix. Le service génère un token JWT qui autorise l'accès à votre boîte Gmail.

> **Sécurité :** Ne commitez jamais `mcp.json` ni `.env` dans votre dépôt Git.

---

## 7. Configuration des modèles IA

Le système est **model-agnostic** : chaque agent peut utiliser un provider et un modèle différents, configurés dans `.env`.

### Format

```
<AGENT>_MODEL=<provider>/<model-name>
```

### Providers supportés

| Provider  | Format                         | Variable d'env requise | Installation                         |
| --------- | ------------------------------ | ---------------------- | ------------------------------------ |
| OpenAI    | `openai/gpt-4o`                | `OPENAI_API_KEY`       | inclus par défaut                    |
| Anthropic | `anthropic/claude-opus-4-6`    | `ANTHROPIC_API_KEY`    | `pip install langchain-anthropic`    |
| Google    | `google/gemini-2.0-flash`      | `GOOGLE_API_KEY`       | `pip install langchain-google-genai` |
| Mistral   | `mistral/mistral-large-latest` | `MISTRAL_API_KEY`      | `pip install langchain-mistralai`    |
| Ollama    | `ollama/llama3.1`              | aucune (local)         | `pip install langchain-ollama`       |

### Chaîne de priorité

```
<AGENT>_MODEL  →  DEFAULT_MODEL  →  openai/gpt-4o (fallback)
```

### Exemples de configuration `.env`

**Configuration tout OpenAI (défaut) :**

```env
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=openai/gpt-4o
```

**Mix OpenAI + Anthropic :**

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

DEFAULT_MODEL=openai/gpt-4o
COMPOSER_MODEL=anthropic/claude-opus-4-6    # meilleure rédaction
REVIEWER_MODEL=anthropic/claude-sonnet-4-6  # bon rapport qualité/coût
INBOX_SCANNER_MODEL=openai/gpt-4o-mini      # tâche simple, modèle léger
```

**Entièrement local avec Ollama :**

```env
DEFAULT_MODEL=ollama/llama3.1
COMPOSER_MODEL=ollama/mistral-nemo
```

**Multi-providers :**

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

COORDINATOR_MODEL=openai/gpt-4o             # raisonnement fort pour le routing
INBOX_SCANNER_MODEL=google/gemini-2.0-flash # rapide pour le scan
COMPOSER_MODEL=openai/gpt-4o               # qualité de rédaction
REVIEWER_MODEL=openai/gpt-4o-mini          # économique pour la révision
```

---

## 8. Lancer l'application

```bash
python main.py
```

**Si des emails non lus sont présents :**

```
============================================================
  Gmail Executive Assistant
============================================================

  [Coordinator] iter=1 → dispatching inbox_scanner
  ...
```

**Si la boîte est vide :**

```
============================================================
  Gmail Executive Assistant
============================================================

  [Coordinator] iter=1 → dispatching inbox_scanner

  [Scanner]     No unread emails found.

============================================================
  No unread emails found. Nothing to do.
============================================================
```

---

## 9. Interaction avec le terminal

### Approbation d'un appel outil

À chaque fois qu'un agent veut accéder à Gmail, le programme se met en pause :

```
------------------------------------------------------------
[Tool Approval Request]
------------------------------------------------------------
  Agent: inbox_scanner
  Task : inbox_scanner
  Tool : list_emails
  Args : {"unreadOnly": true, "limit": 1}

  Allow? [y/n]:
```

- Tapez `y` + Entrée pour autoriser
- Tapez `n` + Entrée pour refuser (une raison peut être fournie)

### Révision du brouillon

Une fois la réponse rédigée et validée par le reviewer IA :

```
------------------------------------------------------------
[Draft Review]
------------------------------------------------------------

Bonjour Marie,

Je serais ravi de vous retrouver mercredi à 14h pour discuter
de l'avancement du projet Alpha. Je confirme ma disponibilité.

Cordialement,
[Votre nom]

------------------------------------------------------------

  [a] Approve and send
  [e] Edit the draft
  [r] Reject — re-draft

Your choice (a/e/r):
```

**Option `a` — Approuver :** L'email est envoyé immédiatement.

**Option `e` — Modifier :** Vous réécrivez le brouillon dans le terminal :

```
Paste revised draft (finish with a line containing only '---'):
> Bonjour Marie, ...
> ---
Comment (Enter to skip): Ajout de la signature complète
```

**Option `r` — Rejeter :** L'agent repart de zéro. Vous pouvez indiquer la raison :

```
Rejection reason (Enter to skip): Le ton est trop formel pour ce contact
```

### Résumé final

```
============================================================
  Status       : sent
  Iterations   : 3
  Feedback     : Approved
============================================================
```

---

## 10. Concepts clés

### Human-in-the-Loop avec `interrupt()`

LangGraph permet de **mettre en pause** l'exécution du graphe à n'importe quel nœud via `interrupt(payload)`. L'état complet est sauvegardé en mémoire (`MemorySaver`). L'exécution reprend exactement là où elle s'était arrêtée en passant `Command(resume=value)`.

C'est ce mécanisme qui permet à l'application d'attendre la validation de l'utilisateur sans bloquer le thread ni perdre le contexte.

### État partagé (`AgentState`)

Tous les agents partagent un état global unique (`AgentState`). Chaque spécialiste lit les champs dont il a besoin et écrit uniquement dans le(s) champ(s) qui lui sont attribués :

| Champ                | Propriétaire                |
| -------------------- | --------------------------- |
| `email_content`      | `inbox_scanner`             |
| `inbox_results`      | `inbox_scanner`             |
| `email_attachments`  | `inbox_scanner`             |
| `attachment_summary` | `attachment_analyzer`       |
| `thread_context`     | `thread_researcher`         |
| `sender_profile`     | `sender_profiler`           |
| `draft_reply`        | `composer` / `human_review` |
| `review_notes`       | `reviewer`                  |
| `next_action`        | `coordinator`               |
| `status`             | chaque spécialiste          |

### Séparation lecture / écriture Gmail

- `get_readonly_gmail_tools()` — expose uniquement `list_emails` et `get_email`. Utilisé par tous les agents sauf l'exécuteur.
- `get_gmail_tools()` — expose en plus `reply_to_email` et `mark_email_as_read`. Utilisé uniquement par `mcp_executor_node`.
- `composer` et `reviewer` n'ont **aucun** accès Gmail (`use_tools=False`).

Cette séparation garantit qu'aucun agent spécialiste ne peut envoyer ou modifier des emails par accident.

### Extraction robuste du message ID

Le serveur MCP peut retourner les identifiants Gmail dans des formats variés. `email_utils.py` tente 4 patterns en cascade :

1. `ID: <valeur>` (format texte plat)
2. `"id": "<valeur>"` (format JSON)
3. `messageId: "<valeur>"` (format YAML/JSON headers)
4. Tout UUID dans le contenu, **en excluant explicitement** les IDs LangChain (préfixe `lc_`)

### Le protocole MCP (Model Context Protocol)

Le serveur MCP est une **passerelle HTTP** entre l'agent Python et l'API Gmail. Il reçoit des appels d'outils standardisés (`list_emails`, `get_email`, etc.) et les traduit en requêtes Gmail API. La connexion est configurée dans `mcp.json` et gérée par `langchain-mcp-adapters`.
