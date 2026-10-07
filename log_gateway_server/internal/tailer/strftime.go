package tailer

import (
	"fmt"
	"regexp"
	"strings"
)

// strftimeTokens maps strftime directives to the equivalent regex fragment.
var strftimeTokens = map[byte]string{
	'Y': `\d{4}`, // 4-digit year
	'm': `\d{2}`, // 2-digit month
	'd': `\d{2}`, // 2-digit day
	'H': `\d{2}`, // 2-digit hour (00-23)
	'M': `\d{2}`, // 2-digit minute
	'S': `\d{2}`, // 2-digit second
	'f': `\d{6}`, // microseconds
	'z': `[+-]\d{4}`,
	'Z': `[A-Za-z_]+`,
	'j': `\d{3}`,
	's': `\d+`,
}

// strftimeToRegex converts a strftime-style format string into a regexp that
// matches the timestamp prefix at the START of a line. Literal characters are
// escaped; %% expands to a literal percent.
func strftimeToRegex(format string) (*regexp.Regexp, error) {
	var b strings.Builder
	b.WriteString(`^`)
	for i := 0; i < len(format); i++ {
		c := format[i]
		if c != '%' {
			b.WriteString(regexp.QuoteMeta(string(c)))
			continue
		}
		if i+1 >= len(format) {
			return nil, fmt.Errorf("trailing %% in prefix_format")
		}
		i++
		tok := format[i]
		if tok == '%' {
			b.WriteString(`%`)
			continue
		}
		re, ok := strftimeTokens[tok]
		if !ok {
			return nil, fmt.Errorf("unsupported strftime directive %%%c", tok)
		}
		b.WriteString(re)
	}
	return regexp.Compile(b.String())
}
