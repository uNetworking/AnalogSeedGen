#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <sstream>

struct Point {
    double x, y;
};

// BIP-39 wordlist (same as provided in HTML)
const std::vector<std::string> bip39 = {
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", "absurd", "abuse",
    "access", "accident", "account", "accuse", "achieve", "acid", "acoustic", "acquire", "across", "act",
    "action", "actor", "actress", "actual", "adapt", "add", "addict", "address", "adjust", "admit",
    "adult", "advance", "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
    "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album", "alcohol", "alert",
    "alien", "all", "alley", "allow", "almost", "alone", "alpha", "already", "also", "alter",
    "always", "amateur", "amazing", "among", "amount", "amused", "analyst", "anchor", "ancient", "anger",
    "angle", "angry", "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
    "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april", "arch", "arctic",
    "area", "arena", "argue", "arm", "armed", "armor", "army", "around", "arrange", "arrest",
    "arrive", "arrow", "art", "artefact", "artist", "artwork", "ask", "aspect", "assault", "asset",
    "assist", "assume", "asthma", "athlete", "atom", "attack", "attend", "attitude", "attract", "auction",
    "audit", "august", "aunt", "author", "auto", "autumn", "average", "avocado", "avoid", "awake",
    "aware", "away", "awesome", "awful", "awkward", "axis", "baby", "bachelor", "bacon", "badge",
    "bag", "balance", "balcony", "ball", "bamboo", "banana", "banner", "bar", "barely", "bargain",
    "barrel", "base", "basic", "basket", "battle", "beach", "bean", "beauty", "because", "become",
    "beef", "before", "begin", "behave", "behind", "believe", "below", "belt", "bench", "benefit",
    "best", "betray", "better", "between", "beyond", "bicycle", "bid", "bike", "bind", "biology",
    "bird", "birth", "bitter", "black", "blade", "blame", "blanket", "blast", "bleak", "bless",
    "blind", "blood", "blossom", "blouse", "blue", "blur", "blush", "board", "boat", "body",
    "boil", "bomb", "bone", "bonus", "book", "boost", "border", "boring", "borrow", "boss",
    "bottom", "bounce", "box", "boy", "bracket", "brain", "brand", "brass", "brave", "bread",
    "breeze", "brick", "bridge", "brief", "bright", "bring", "brisk", "broccoli", "broken", "bronze",
    "broom", "brother", "brown", "brush", "bubble", "buddy", "budget", "buffalo", "build", "bulb",
    "bulk", "bullet", "bundle", "bunker", "burden", "burger", "burst", "bus", "business", "busy",
    "butter", "buyer", "buzz", "cabbage", "cabin", "cable", "cactus", "cage", "cake", "call",
    "calm", "camera", "camp", "can", "canal", "cancel", "candy", "cannon", "canoe", "canvas",
    "canyon", "capable", "capital", "captain", "car", "carbon", "card", "cargo", "carpet", "carry",
    "cart", "case", "cash", "casino", "castle", "casual", "cat", "catalog", "catch", "category",
    "cattle", "caught", "cause", "caution", "cave", "ceiling", "celery", "cement", "census", "century",
    "cereal", "certain", "chair", "chalk", "champion", "change", "chaos", "chapter", "charge", "chase",
    "chat", "cheap", "check", "cheese", "chef", "cherry", "chest", "chicken", "chief", "child",
    "chimney", "choice", "choose", "chronic", "chuckle", "chunk", "churn", "cigar", "cinnamon", "circle",
    "citizen", "city", "civil", "claim", "clap", "clarify", "claw", "clay", "clean", "clerk",
    "clever", "click", "client", "cliff", "climb", "clinic", "clip", "clock", "clog", "close",
    "cloth", "cloud", "clown", "club", "clump", "cluster", "clutch", "coach", "coast", "coconut",
    "code", "coffee", "coil", "coin", "collect", "color", "column", "combine", "come", "comfort",
    "comic", "common", "company", "concert", "conduct", "confirm", "congress", "connect", "consider", "control",
    "convince", "cook", "cool", "copper", "copy", "coral", "core", "corn", "correct", "cost",
    "cotton", "couch", "country", "couple", "course", "cousin", "cover", "coyote", "crack", "cradle",
    "craft", "cram", "crane", "crash", "crater", "crawl", "crazy", "cream", "credit", "creek",
    "crew", "cricket", "crime", "crisp", "critic", "crop", "cross", "crouch", "crowd", "crucial",
    "cruel", "cruise", "crumble", "crunch", "crush", "cry", "crystal", "cube", "culture", "cup",
    "cupboard", "curious", "current", "curtain", "curve", "cushion", "custom", "cute", "cycle", "dad",
    "damage", "damp", "dance", "danger", "daring", "dash", "daughter", "dawn", "day", "deal",
    "debate", "debris", "decade", "december", "decide", "decline", "decorate", "decrease", "deer", "defense",
    "define", "defy", "degree", "delay", "deliver", "demand", "demise", "denial", "dentist", "deny",
    "depart", "depend", "deposit", "depth", "deputy", "derive", "describe", "desert", "design", "desk",
    "despair", "destroy", "detail", "detect", "develop", "device", "devote", "diagram", "dial", "diamond",
    "diary", "dice", "diesel", "diet", "differ", "digital", "dignity", "dilemma", "dinner", "dinosaur",
    "direct", "dirt", "disagree", "discover", "disease", "dish", "dismiss", "disorder", "display", "distance",
    "divert", "divide", "divorce", "dizzy", "doctor", "document", "dog", "doll", "dolphin", "domain",
    "donate", "donkey", "donor", "door", "dose", "double", "dove", "draft", "dragon", "drama",
    "drastic", "draw", "dream", "dress", "drift", "drill", "drink", "drip", "drive", "drop",
    "drum", "dry", "duck", "dumb", "dune", "during", "dust", "dutch", "duty", "dwarf",
    "dynamic", "eager", "eagle", "early", "earn", "earth", "easily", "east", "easy", "echo",
    "ecology", "economy", "edge", "edit", "educate", "effort", "egg", "eight", "either", "elbow",
    "elder", "electric", "elegant", "element", "elephant", "elevator", "elite", "else", "embark", "embody",
    "embrace", "emerge", "emotion", "employ", "empower", "empty", "enable", "enact", "end", "endless",
    "endorse", "enemy", "energy", "enforce", "engage", "engine", "enhance", "enjoy", "enlist", "enough",
    "enrich", "enroll", "ensure", "enter", "entire", "entry", "envelope", "episode", "equal", "equip",
    "era", "erase", "erode", "erosion", "error", "erupt", "escape", "essay", "essence", "estate",
    "eternal", "ethics", "evidence", "evil", "evoke", "evolve", "exact", "example", "excess", "exchange",
    "excite", "exclude", "excuse", "execute", "exercise", "exhaust", "exhibit", "exile", "exist", "exit",
    "exotic", "expand", "expect", "expire", "explain", "expose", "express", "extend", "extra", "eye",
    "eyebrow", "fabric", "face", "faculty", "fade", "faint", "faith", "fall", "false", "fame",
    "family", "famous", "fan", "fancy", "fantasy", "farm", "fashion", "fat", "fatal", "father",
    "fatigue", "fault", "favorite", "feature", "february", "federal", "fee", "feed", "feel", "female",
    "fence", "festival", "fetch", "fever", "few", "fiber", "fiction", "field", "figure", "file",
    "film", "filter", "final", "find", "fine", "finger", "finish", "fire", "firm", "first",
    "fiscal", "fish", "fit", "fitness", "fix", "flag", "flame", "flash", "flat", "flavor",
    "flee", "flight", "flip", "float", "flock", "floor", "flower", "fluid", "flush", "fly",
    "foam", "focus", "fog", "foil", "fold", "follow", "food", "foot", "force", "forest",
    "forget", "fork", "fortune", "forum", "forward", "fossil", "foster", "found", "fox", "fragile",
    "frame", "frequent", "fresh", "friend", "fringe", "frog", "front", "frost", "frown", "frozen",
    "fruit", "fuel", "fun", "funny", "furnace", "fury", "future", "gadget", "gain", "galaxy",
    "gallery", "game", "gap", "garage", "garbage", "garden", "garlic", "garment", "gas", "gasp",
    "gate", "gather", "gauge", "gaze", "general", "genius", "genre", "gentle", "genuine", "gesture",
    "ghost", "giant", "gift", "giggle", "ginger", "giraffe", "girl", "give", "glad", "glance",
    "glare", "glass", "glide", "glimpse", "globe", "gloom", "glory", "glove", "glow", "glue",
    "goat", "goddess", "gold", "good", "goose", "gorilla", "gospel", "gossip", "govern", "gown",
    "grab", "grace", "grain", "grant", "grape", "grass", "gravity", "great", "green", "grid",
    "grief", "grit", "grocery", "group", "grow", "grunt", "guard", "guess", "guide", "guilt",
    "guitar", "gun", "gym", "habit", "hair", "half", "hammer", "hamster", "hand", "happy",
    "harbor", "hard", "harsh", "harvest", "hat", "have", "hawk", "hazard", "head", "health",
    "heart", "heavy", "hedgehog", "height", "hello", "helmet", "help", "hen", "hero", "hidden",
    "high", "hill", "hint", "hip", "hire", "history", "hobby", "hockey", "hold", "hole",
    "holiday", "hollow", "home", "honey", "hood", "hope", "horn", "horror", "horse", "hospital",
    "host", "hotel", "hour", "hover", "hub", "huge", "human", "humble", "humor", "hundred",
    "hungry", "hunt", "hurdle", "hurry", "hurt", "husband", "hybrid", "ice", "icon", "idea",
    "identify", "idle", "ignore", "ill", "illegal", "illness", "image", "imitate", "immense", "immune",
    "impact", "impose", "improve", "impulse", "inch", "include", "income", "increase", "index", "indicate",
    "indoor", "industry", "infant", "inflict", "inform", "inhale", "inherit", "initial", "inject", "injury",
    "inmate", "inner", "innocent", "input", "inquiry", "insane", "insect", "inside", "inspire", "install",
    "intact", "interest", "into", "invest", "invite", "involve", "iron", "island", "isolate", "issue",
    "item", "ivory", "jacket", "jaguar", "jar", "jazz", "jealous", "jeans", "jelly", "jewel",
    "job", "join", "joke", "journey", "joy", "judge", "juice", "jump", "jungle", "junior",
    "junk", "just", "kangaroo", "keen", "keep", "ketchup", "key", "kick", "kid", "kidney",
    "kind", "kingdom", "kiss", "kit", "kitchen", "kite", "kitten", "kiwi", "knee", "knife",
    "knock", "know", "lab", "label", "labor", "ladder", "lady", "lake", "lamp", "language",
    "laptop", "large", "later", "latin", "laugh", "laundry", "lava", "law", "lawn", "lawsuit",
    "layer", "lazy", "leader", "leaf", "learn", "leave", "lecture", "left", "leg", "legal",
    "legend", "leisure", "lemon", "lend", "length", "lens", "leopard", "lesson", "letter", "level",
    "liar", "liberty", "library", "license", "life", "lift", "light", "like", "limb", "limit",
    "link", "lion", "liquid", "list", "little", "live", "lizard", "load", "loan", "lobster",
    "local", "lock", "logic", "lonely", "long", "loop", "lottery", "loud", "lounge", "love",
    "loyal", "lucky", "luggage", "lumber", "lunar", "lunch", "luxury", "lyrics", "machine", "mad",
    "magic", "magnet", "maid", "mail", "main", "major", "make", "mammal", "man", "manage",
    "mandate", "mango", "mansion", "manual", "maple", "marble", "march", "margin", "marine", "market",
    "marriage", "mask", "mass", "master", "match", "material", "math", "matrix", "matter", "maximum",
    "maze", "meadow", "mean", "measure", "meat", "mechanic", "medal", "media", "melody", "melt",
    "member", "memory", "mention", "menu", "mercy", "merge", "merit", "merry", "mesh", "message",
    "metal", "method", "middle", "midnight", "milk", "million", "mimic", "mind", "minimum", "minor",
    "minute", "miracle", "mirror", "misery", "miss", "mistake", "mix", "mixed", "mixture", "mobile",
    "model", "modify", "mom", "moment", "monitor", "monkey", "monster", "month", "moon", "moral",
    "more", "morning", "mosquito", "mother", "motion", "motor", "mountain", "mouse", "move", "movie",
    "much", "muffin", "mule", "multiply", "muscle", "museum", "mushroom", "music", "must", "mutual",
    "myself", "mystery", "myth", "naive", "name", "napkin", "narrow", "nasty", "nation", "nature",
    "near", "neck", "need", "negative", "neglect", "neither", "nephew", "nerve", "nest", "net",
    "network", "neutral", "never", "news", "next", "nice", "night", "noble", "noise", "nominee",
    "noodle", "normal", "north", "nose", "notable", "note", "nothing", "notice", "novel", "now",
    "nuclear", "number", "nurse", "nut", "oak", "obey", "object", "oblige", "obscure", "observe",
    "obtain", "obvious", "occur", "ocean", "october", "odor", "off", "offer", "office", "often",
    "oil", "okay", "old", "olive", "olympic", "omit", "once", "one", "onion", "online",
    "only", "open", "opera", "opinion", "oppose", "option", "orange", "orbit", "orchard", "order",
    "ordinary", "organ", "orient", "original", "orphan", "ostrich", "other", "outdoor", "outer", "output",
    "outside", "oval", "oven", "over", "own", "owner", "oxygen", "oyster", "ozone", "pact",
    "paddle", "page", "pair", "palace", "palm", "panda", "panel", "panic", "panther", "paper",
    "parade", "parent", "park", "parrot", "party", "pass", "patch", "path", "patient", "patrol",
    "pattern", "pause", "pave", "payment", "peace", "peanut", "pear", "peasant", "pelican", "pen",
    "penalty", "pencil", "people", "pepper", "perfect", "permit", "person", "pet", "phone", "photo",
    "phrase", "physical", "piano", "picnic", "picture", "piece", "pig", "pigeon", "pill", "pilot",
    "pink", "pioneer", "pipe", "pistol", "pitch", "pizza", "place", "planet", "plastic", "plate",
    "play", "please", "pledge", "pluck", "plug", "plunge", "poem", "poet", "point", "polar",
    "pole", "police", "pond", "pony", "pool", "popular", "portion", "position", "possible", "post",
    "potato", "pottery", "poverty", "powder", "power", "practice", "praise", "predict", "prefer", "prepare",
    "present", "pretty", "prevent", "price", "pride", "primary", "print", "priority", "prison", "private",
    "prize", "problem", "process", "produce", "profit", "program", "project", "promote", "proof", "property",
    "prosper", "protect", "proud", "provide", "public", "pudding", "pull", "pulp", "pulse", "pumpkin",
    "punch", "pupil", "puppy", "purchase", "purity", "purpose", "purse", "push", "put", "puzzle",
    "pyramid", "quality", "quantum", "quarter", "question", "quick", "quit", "quiz", "quote", "rabbit",
    "raccoon", "race", "rack", "radar", "radio", "rail", "rain", "raise", "rally", "ramp",
    "ranch", "random", "range", "rapid", "rare", "rate", "rather", "raven", "raw", "razor",
    "ready", "real", "reason", "rebel", "rebuild", "recall", "receive", "recipe", "record", "recycle",
    "reduce", "reflect", "reform", "refuse", "region", "regret", "regular", "reject", "relax", "release",
    "relief", "rely", "remain", "remember", "remind", "remove", "render", "renew", "rent", "reopen",
    "repair", "repeat", "replace", "report", "require", "rescue", "resemble", "resist", "resource", "response",
    "result", "retire", "retreat", "return", "reunion", "reveal", "review", "reward", "rhythm", "rib",
    "ribbon", "rice", "rich", "ride", "ridge", "rifle", "right", "rigid", "ring", "riot",
    "ripple", "risk", "ritual", "rival", "river", "road", "roast", "robot", "robust", "rocket",
    "romance", "roof", "rookie", "room", "rose", "rotate", "rough", "round", "route", "royal",
    "rubber", "rude", "rug", "rule", "run", "runway", "rural", "sad", "saddle", "sadness",
    "safe", "sail", "salad", "salmon", "salon", "salt", "salute", "same", "sample", "sand",
    "satisfy", "satoshi", "sauce", "sausage", "save", "say", "scale", "scan", "scare", "scatter",
    "scene", "scheme", "school", "science", "scissors", "scorpion", "scout", "scrap", "screen", "script",
    "scrub", "sea", "search", "season", "seat", "second", "secret", "section", "security", "seed",
    "seek", "segment", "select", "sell", "seminar", "senior", "sense", "sentence", "series", "service",
    "session", "settle", "setup", "seven", "shadow", "shaft", "shallow", "share", "shed", "shell",
    "sheriff", "shield", "shift", "shine", "ship", "shiver", "shock", "shoe", "shoot", "shop",
    "short", "shoulder", "shove", "shrimp", "shrug", "shuffle", "shy", "sibling", "sick", "side",
    "siege", "sight", "sign", "silent", "silk", "silly", "silver", "similar", "simple", "since",
    "sing", "siren", "sister", "situate", "six", "size", "skate", "sketch", "ski", "skill",
    "skin", "skirt", "skull", "slab", "slam", "sleep", "slender", "slice", "slide", "slight",
    "slim", "slogan", "slot", "slow", "slush", "small", "smart", "smile", "smoke", "smooth",
    "snack", "snake", "snap", "sniff", "snow", "soap", "soccer", "social", "sock", "soda",
    "soft", "solar", "soldier", "solid", "solution", "solve", "someone", "song", "soon", "sorry",
    "sort", "soul", "sound", "soup", "source", "south", "space", "spare", "spatial", "spawn",
    "speak", "special", "speed", "spell", "spend", "sphere", "spice", "spider", "spike", "spin",
    "spirit", "split", "spoil", "sponsor", "spoon", "sport", "spot", "spray", "spread", "spring",
    "spy", "square", "squeeze", "squirrel", "stable", "stadium", "staff", "stage", "stairs", "stamp",
    "stand", "start", "state", "stay", "steak", "steel", "stem", "step", "stereo", "stick",
    "still", "sting", "stock", "stomach", "stone", "stool", "story", "stove", "strategy", "street",
    "strike", "strong", "struggle", "student", "stuff", "stumble", "style", "subject", "submit", "subway",
    "success", "such", "sudden", "suffer", "sugar", "suggest", "suit", "summer", "sun", "sunny",
    "sunset", "super", "supply", "supreme", "sure", "surface", "surge", "surprise", "surround", "survey",
    "suspect", "sustain", "swallow", "swamp", "swap", "swarm", "swear", "sweet", "swift", "swim",
    "swing", "switch", "sword", "symbol", "symptom", "syrup", "system", "table", "tackle", "tag",
    "tail", "talent", "talk", "tank", "tape", "target", "task", "taste", "tattoo", "taxi",
    "teach", "team", "tell", "ten", "tenant", "tennis", "tent", "term", "test", "text",
    "thank", "that", "theme", "then", "theory", "there", "they", "thing", "this", "thought",
    "three", "thrive", "throw", "thumb", "thunder", "ticket", "tide", "tiger", "tilt", "timber",
    "time", "tiny", "tip", "tired", "tissue", "title", "toast", "tobacco", "today", "toddler",
    "toe", "together", "toilet", "token", "tomato", "tomorrow", "tone", "tongue", "tonight", "tool",
    "tooth", "top", "topic", "topple", "torch", "tornado", "tortoise", "toss", "total", "tourist",
    "toward", "tower", "town", "toy", "track", "trade", "traffic", "tragic", "train", "transfer",
    "trap", "trash", "travel", "tray", "treat", "tree", "trend", "trial", "tribe", "trick",
    "trigger", "trim", "trip", "trophy", "trouble", "truck", "true", "truly", "trumpet", "trust",
    "truth", "try", "tube", "tuition", "tumble", "tuna", "tunnel", "turkey", "turn", "turtle",
    "twelve", "twenty", "twice", "twin", "twist", "two", "type", "typical", "ugly", "umbrella",
    "unable", "unaware", "uncle", "uncover", "under", "undo", "unfair", "unfold", "unhappy", "uniform",
    "unique", "unit", "universe", "unknown", "unlock", "until", "unusual", "unveil", "update", "upgrade",
    "uphold", "upon", "upper", "upset", "urban", "urge", "usage", "use", "used", "useful",
    "useless", "usual", "utility", "vacant", "vacuum", "vague", "valid", "valley", "valve", "van",
    "vanish", "vapor", "various", "vast", "vault", "vehicle", "velvet", "vendor", "venture", "venue",
    "verb", "verify", "version", "very", "vessel", "veteran", "viable", "vibrant", "vicious", "victory",
    "video", "view", "village", "vintage", "violin", "virtual", "virus", "visa", "visit", "visual",
    "vital", "vivid", "vocal", "voice", "void", "volcano", "volume", "vote", "voyage", "wage",
    "wagon", "wait", "walk", "wall", "walnut", "want", "warfare", "warm", "warrior", "wash",
    "wasp", "waste", "water", "wave", "way", "wealth", "weapon", "wear", "weasel", "weather",
    "web", "wedding", "weekend", "weird", "welcome", "west", "wet", "whale", "what", "wheat",
    "wheel", "when", "where", "whip", "whisper", "wide", "width", "wife", "wild", "will",
    "win", "window", "wine", "wing", "wink", "winner", "winter", "wire", "wisdom", "wise",
    "wish", "witness", "wolf", "woman", "wonder", "wood", "wool", "word", "work", "world",
    "worry", "worth", "wrap", "wreck", "wrestle", "wrist", "write", "wrong", "yard", "year",
    "yellow", "you", "young", "youth", "zebra", "zero", "zone", "zoo"
};

// Replace "word" nodes with BIP-39 words based on pageOffset
std::vector<std::string> replaceBip39Words(const std::vector<std::string>& nodes, int pageOffset) {
    const int startIndex = 64 * pageOffset;
    const int wordCount = 64;
    
    if (startIndex + wordCount > static_cast<int>(bip39.size())) {
        throw std::runtime_error("Requested range exceeds BIP-39 wordlist length");
    }
    
    std::vector<std::string> updatedNodes = nodes;
    int bip39Index = startIndex;
    for (size_t i = 0; i < updatedNodes.size() && bip39Index < startIndex + wordCount; ++i) {
        if (updatedNodes[i] == "word") {
            updatedNodes[i] = bip39[bip39Index] + " (" + std::to_string(bip39Index + 1) + ")  ";
            ++bip39Index;
        }
    }
    
    return updatedNodes;
}

// Calculate node positions and bounds
struct TreeLayout {
    std::vector<Point> positions;
    double minX, maxX, minY, maxY;
};

TreeLayout calculatePositions(const std::vector<std::string>& nodes, const double nodeRadius = 20.0, const double levelHeight = 100.0, const double siblingDistance = 600.0, const double yOffset = 100.0) {
    TreeLayout layout;
    layout.positions.resize(nodes.size(), {0.0, 0.0});
    layout.minX = layout.minY = std::numeric_limits<double>::infinity();
    layout.maxX = layout.maxY = -std::numeric_limits<double>::infinity();
    
    auto assignPositions = [&](int i, int d, double x, auto& self) -> double {
        if (i >= static_cast<int>(nodes.size())) return 0.0;
        int leftChild = 2 * i + 1;
        int rightChild = 2 * i + 2;
        double leftWidth = 0.0, rightWidth = 0.0;
        double leftX = x, rightX = x;
        
        if (leftChild < static_cast<int>(nodes.size())) {
            leftWidth = self(leftChild, d + 1, x - siblingDistance / std::pow(2.0, d), self);
            leftX = x - siblingDistance / std::pow(2.0, d);
        }
        
        if (rightChild < static_cast<int>(nodes.size())) {
            rightWidth = self(rightChild, d + 1, x + siblingDistance / std::pow(2.0, d), self);
            rightX = x + siblingDistance / std::pow(2.0, d);
        }
        
        double nodeX = (leftWidth > 0 && rightWidth > 0) ? (leftX + rightX) / 2.0 : x;
        double nodeY = d * levelHeight + nodeRadius + yOffset;
        layout.positions[i] = {nodeX, nodeY};
        
        layout.minX = std::min(layout.minX, nodeX + 400.0 - nodeRadius);
        layout.maxX = std::max(layout.maxX, nodeX + 400.0 + nodeRadius);
        layout.minY = std::min(layout.minY, nodeY - nodeRadius);
        layout.maxY = std::max(layout.maxY, nodeY + nodeRadius);

        return std::max(std::max(leftWidth, rightWidth), siblingDistance / std::pow(2.0, d));
    };
    
    assignPositions(0, 0, 0.0, assignPositions);
    
    if (!layout.positions.empty()) {
        layout.minY = std::min(layout.minY, layout.positions[0].y - 90.0 - nodeRadius);
        layout.maxY = layout.maxY + 200.0;
    }
    
    return layout;
}

// Generate SVG node content
std::string drawNode(double x, double y, const std::string& value, int index, const std::vector<std::string>& nodes) {
    std::ostringstream oss;
    bool isLeaf = (2 * index + 1) >= static_cast<int>(nodes.size()) && (2 * index + 2) >= static_cast<int>(nodes.size());
    
    if (value == "⚄") {
        // Draw die as a rounded square with pips
        oss << "<rect x=\"" << (x + 400.0 - 15.0) << "\" y=\"" << (y + 50.0 - 15.0) << "\" width=\"30\" height=\"30\" rx=\"5\" ry=\"5\" fill=\"#fff\" stroke=\"#000\"/>\n";
        const std::vector<Point> pipPositions = {
            {x + 400.0 - 8.0, y + 50.0 - 8.0},
            {x + 400.0, y + 50.0},
            {x + 400.0 + 8.0, y + 50.0 + 8.0}
        };
        for (const auto& pos : pipPositions) {
            oss << "<circle cx=\"" << pos.x << "\" cy=\"" << pos.y << "\" r=\"3\" fill=\"#000\"/>\n";
        }
    } else {
        // Draw text node
        oss << "<text x=\"" << (x + 400.0) << "\" y=\"" << (y + 40.0) << "\" text-anchor=\"end\" font-size=\"28px\" dominant-baseline=\"central\" fill=\"#000\"";
        if (isLeaf) {
            oss << " transform=\"rotate(-90, " << (x + 400.0) << ", " << (y + 40.0) << ")\"";
        }
        oss << ">" << value << "</text>\n";
    }
    
    return oss.str();
}

// Generate SVG edge content
std::string drawEdge(const Point& parentPos, const Point& childPos, const std::string& label, double nodeRadius) {
    std::ostringstream oss;
    double x1 = parentPos.x + 400.0;
    double y1 = parentPos.y + 50.0;
    double x2Original = childPos.x + 400.0;
    double y2Original = childPos.y + 50.0;
    double dx = x2Original - x1;
    double dy = y2Original - y1;
    double angle = std::atan2(dy, dx);
    double x2 = x2Original - nodeRadius * 1.1 * std::cos(angle);
    double y2 = y2Original - nodeRadius * 1.1 * std::sin(angle);
    
    oss << "<path d=\"M" << x1 << "," << y1 << " L" << x2 << "," << y2 << "\" stroke=\"#000\" stroke-width=\"1\" marker-end=\"url(#arrowhead)\"/>\n";
    
    double midX = (x1 + x2) / 2.0;
    double midY = (y1 + y2) / 2.0;
    double textOffset = -8.0;
    double perpAngle = angle + M_PI / 2.0;
    double offsetX = textOffset * std::cos(perpAngle);
    double offsetY = textOffset * std::sin(perpAngle);
    double textX = midX + (label == "Even" ? -offsetX : offsetX);
    double textY = midY + (label == "Even" ? -offsetY : offsetY);
    double rotationAngle = (angle * 180.0 / M_PI) + (label == "Even" ? 180.0 : 0.0);
    
    oss << "<text x=\"" << textX << "\" y=\"" << textY << "\" text-anchor=\"middle\" font-size=\"16px\" dominant-baseline=\"central\" fill=\"#000\" transform=\"rotate(" << rotationAngle << ", " << textX << ", " << textY << ")\">" << label << "</text>\n";
    
    return oss.str();
}

// Generate the SVG content
std::string generateSVG(const std::vector<std::string>& nodes, int pageOffset) {
    const double nodeRadius = 20.0;
    const double levelHeight = 100.0;
    const double siblingDistance = 600.0;
    const double yOffset = 100.0;
    
    auto layout = calculatePositions(nodes, nodeRadius, levelHeight, siblingDistance, yOffset);
    const auto& positions = layout.positions;
    double padding = 50.0;
    double viewBoxWidth = layout.maxX - layout.minX + 2 * padding;
    double viewBoxHeight = layout.maxY - layout.minY + 2 * padding;
    
    std::ostringstream svg;
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"" << (layout.minX - padding) << " " << (layout.minY - padding) << " " << viewBoxWidth << " " << viewBoxHeight << "\" preserveAspectRatio=\"xMidYMid meet\">\n";
    
    // Define arrowhead marker
    svg << "<defs>\n";
    svg << "<marker id=\"arrowhead\" markerWidth=\"12\" markerHeight=\"8\" refX=\"10\" refY=\"4\" orient=\"auto\">\n";
    svg << "<polygon points=\"0 0, 12 4, 0 8\" fill=\"#000\" stroke=\"#000\" stroke-width=\"0.5\"/>\n";
    svg << "</marker>\n";
    svg << "</defs>\n";
    
    // Draw page number
    svg << "<text x=\"" << (10.0 + layout.minX) << "\" y=\"" << (10.0 + layout.minY) << "\" text-anchor=\"start\" font-size=\"45px\" fill=\"#000\">Continuation sheet " << (pageOffset + 1) << "</text>\n";
    
    // Draw "Start" edge to root
    if (!positions.empty()) {
        Point startPos = {positions[0].x, positions[0].y - 110.0};
        svg << drawEdge(startPos, positions[0], "Continue", nodeRadius);
    }
    
    // Draw edges
    for (size_t i = 0; i < nodes.size(); ++i) {
        int leftChild = 2 * i + 1;
        int rightChild = 2 * i + 2;
        if (leftChild < static_cast<int>(nodes.size()) && positions[leftChild].x != 0.0) {
            svg << drawEdge(positions[i], positions[leftChild], "Even", nodeRadius);
        }
        if (rightChild < static_cast<int>(nodes.size()) && positions[rightChild].x != 0.0) {
            svg << drawEdge(positions[i], positions[rightChild], "Odd", nodeRadius);
        }
    }
    
    // Draw nodes
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (positions[i].x != 0.0 || positions[i].y != 0.0) {
            svg << drawNode(positions[i].x, positions[i].y, nodes[i], i, nodes);
        }
    }
    
    svg << "</svg>\n";
    return svg.str();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <pageOffset>" << std::endl;
        return 1;
    }
    
    int pageOffset;
    try {
        pageOffset = std::stoi(argv[1]) - 1;
        if (pageOffset < 0) {
            throw std::invalid_argument("pageOffset must be non-negative");
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid pageOffset: " << argv[1] << std::endl;
        return 1;
    }
    
    // Generate nodes
    std::vector<std::string> nodes;
    for (int i = 0; i < 6; ++i) {
        int toPushThisLevel = std::pow(2, i);
        for (int j = 0; j < toPushThisLevel; ++j) {
            nodes.push_back("⚄");
        }
    }
    for (int i = 0; i < 64; ++i) {
        nodes.push_back("word");
    }
    
    // Replace BIP-39 words
    try {
        nodes = replaceBip39Words(nodes, pageOffset);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    
    // Generate SVG content
    std::string svgContent = generateSVG(nodes, pageOffset);
    
    // Write to file
    std::string filename = "Page" + std::to_string(pageOffset + 1) + ".svg";
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }
    
    outFile << svgContent;
    outFile.close();
    
    std::cout << "SVG file generated: " << filename << std::endl;
    return 0;
}