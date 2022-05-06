#include <vector>
#include <string>

#ifndef dict_19203u10_fosun
#define dict_19203u10_fosun

class Word
{
    public:
    Word( std::string s );
    std::vector< char > word;
    bool letter_idx_match( std::vector< char > indices );
    bool letter_match( std::vector< char > known_letters);
    bool letter_exclude( std::vector< char > letters_not_in_word );
    bool valid( Info* information );
};

class Info
{
    public:
        std::vector< char > known;       // ordered
        std::vector< char > in_word;     // unordered
        std::vector< char > not_in_word; // unordered

};

// assumes lower case
char lower_case( char c );

#endif