#ifndef dict_19203u10
#define dict_19203u10

#include "dictionary.hpp"
using namespace std;

bool char_in_vector( char c, vector<char>& v )
{
    for( int i(0); i < v.size(); ++i )
    {
        if( v[i] == c )
            return true;
    }
    return false;
}


Word::Word( string s )
{
    vector<char> word( s.size() );
    for( int i(0); i < s.size(); ++i )
    {
        word[i] = s[i];
    }
}

bool Word::letter_idx_match( vector<char> indices )
{
    for( int i(0); i < indices.size(); ++i )
    {
        if( !char_in_vector( indices[i], this->word) )
            return false;
    }
    return true;
}

bool Word::letter_match( vector< char> known_letters )
{
    for( int i(0); i < known_letters.size(); ++i )
    {
        bool letter_found = false;
        for( int j(0); j < this->word.size(); ++j )
        {
            if( word[j] == known_letters[i] )
                letter_found = true;
        }
        if( letter_found == false )
            return false;
    }
    return true;
}

bool Word::letter_exclude( vector<char> letters_not_in_word )
{
    for( int i(0); i < word.size(); ++i )
    {
        if( char_in_vector( word[i], letters_not_in_word )
            return false;
    }
    return true;
}

bool Word::valid( Info* information )
{
    if( )
}
#endif 